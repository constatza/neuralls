"""Dataset persistence composition for generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation.orchestration import build_dataset_payload
from neuralls.domain.generation.ports import DatasetAccumulatorPort
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.storage.dataset_digest import (
    current_dataset_digest,
    dataset_content_digest,
    stat_digest,
)
from neuralls.platform.storage.datasets import (
    GenerationDatasetStorage,
    make_generation_dataset_storage,
)
from neuralls.platform.storage.manifest_io import read_dataset_manifest, save_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat

_DEFAULT_SOLVER = make_solver()
_DEFAULT_SOLVER_OVERRIDES: dict[str, Any] = {
    "residuals": _DEFAULT_SOLVER,
    "gaussian_residuals": _DEFAULT_SOLVER,
    "search_directions": _DEFAULT_SOLVER,
}


def _guard_format_conflict(dataset_dir: Path, intended: DatasetFormat) -> None:
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    if not manifest_path.exists():
        return
    try:
        manifest = read_dataset_manifest(dataset_dir)
    except FileNotFoundError, ValueError:
        return
    existing = manifest.matrix.format
    if existing != intended:
        raise ValueError(
            f"Dataset at '{dataset_dir}' was generated as format '{existing}'. "
            f"Cannot overwrite with format '{intended}'. "
            f"Delete the existing dataset directory first."
        )


def _stamp_dataset_identity(dataset_dir: Path, identity: StageIdentity | None) -> None:
    """Record digests (and identity, when known) of the freshly written dataset.

    Runs after the storage backend has written both the arrays and the
    manifest. Until this stamp lands the manifest has no ``identity_key``, so a
    crash in between leaves a dataset that is regenerated, never trusted.

    Args:
        dataset_dir: Directory holding the dataset just written.
        identity: Generation identity to persist, or None for identity-less builds.
    """
    manifest = read_dataset_manifest(dataset_dir)
    stamped = replace(
        manifest,
        content_digest=dataset_content_digest(dataset_dir),
        stat_digest=stat_digest(dataset_dir),
        identity_key=identity.key if identity is not None else None,
        identity_components=dict(identity.components) if identity is not None else None,
    )
    save_dataset_manifest(dataset_dir, stamped)


def is_dataset_reusable(dataset_dir: Path, identity: StageIdentity) -> bool:
    """Return True when dataset_dir holds a dataset generated under ``identity``.

    Requires the manifest to carry the same ``identity_key`` (a manifest without
    one is never trusted) and the artifacts' current content digest to equal the
    stamped one. Read-only.

    Args:
        dataset_dir: Candidate dataset directory.
        identity: Identity the dataset would be generated under now.

    Returns:
        True when regeneration can be skipped.
    """
    try:
        manifest = read_dataset_manifest(dataset_dir)
        if manifest.identity_key != identity.key or manifest.content_digest is None:
            return False
        return current_dataset_digest(dataset_dir) == manifest.content_digest
    except FileNotFoundError, ValueError:
        return False


def _with_default_solvers(spec: DatasetSpec) -> DatasetSpec:
    """Layer the composition-provided default tracing solvers under the caller's."""
    mixture = spec.mixture
    return replace(
        spec,
        mixture=replace(
            mixture,
            solver_overrides={**_DEFAULT_SOLVER_OVERRIDES, **(mixture.solver_overrides or {})},
        ),
    )


def build_dataset(
    source: SourceSpec,
    spec: DatasetSpec,
    dataset_dir: str,
    *,
    dataset_format: DatasetFormat = "hdf5",
    storage: GenerationDatasetStorage | None = None,
    accumulator: DatasetAccumulatorPort | None = None,
    force: bool = False,
    identity: StageIdentity | None = None,
) -> str:
    """Build a persisted dataset by composing domain payload generation with storage.

    Skips regeneration when `dataset_dir` holds a dataset stamped with the same
    generation `identity` whose artifact content still matches its stamped
    digest, unless `force=True`. Without an `identity` nothing can be proven
    about an existing dataset, so it is always regenerated. Each fresh write
    stamps the identity and content/stat digests into the manifest.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: How the dataset is assembled — strategy budgets, RNG controls,
            replacement policy and normalization.
        dataset_dir: Target directory for the persisted dataset.
        dataset_format: Storage format family for the persisted artifacts.
        storage: Optional storage override; defaults to the format's storage.
        accumulator: Optional accumulator override; defaults to the storage's.
        force: Regenerate even when a complete dataset already exists.
        identity: Generation identity used for reuse and stamped into the manifest.

    Returns:
        The `dataset_dir` that now holds the dataset.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    _guard_format_conflict(dataset_path, dataset_format)
    if not force and identity is not None and is_dataset_reusable(dataset_path, identity):
        logger.info(f"Dataset already exists at '{dataset_dir}'; skipping regeneration.")
        return dataset_dir
    dataset_storage = storage or make_generation_dataset_storage(dataset_format)
    acc: DatasetAccumulatorPort = accumulator or dataset_storage.make_accumulator(dataset_path)
    payload = build_dataset_payload(source, _with_default_solvers(spec), accumulator=acc)
    dataset_storage.write_dataset(dataset_path, payload)
    _stamp_dataset_identity(dataset_path, identity)
    return dataset_dir
