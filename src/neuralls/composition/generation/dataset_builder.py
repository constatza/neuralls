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
from neuralls.platform.caching import compute_dataset_fingerprint
from neuralls.platform.storage.dataset_readers import DatasetArtifacts, resolve_dataset_artifacts
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


def _fingerprinted_artifact_paths(artifacts: DatasetArtifacts) -> tuple[Path, Path, Path]:
    """Return the artifacts that identify a dataset, in the training-side order.

    The training reuse-check fingerprints exactly these three artifacts, so
    generation must fingerprint the same files resolved the same way for both
    stages to agree on whether a dataset has changed.

    Args:
        artifacts: Manifest-resolved dataset artifacts.

    Returns:
        The matrix, RHS and solutions artifact paths.
    """
    return (artifacts.matrix.path, artifacts.rhs.path, artifacts.solutions.path)


def _stamp_dataset_fingerprint(dataset_dir: Path) -> None:
    """Record a fingerprint of the freshly written artifacts in the manifest.

    Runs after the storage backend has written both the arrays and the
    manifest, so the artifact paths can be resolved from the manifest exactly
    as every reader resolves them.

    Args:
        dataset_dir: Directory holding the dataset just written.
    """
    artifacts = resolve_dataset_artifacts(dataset_dir)
    fingerprint = compute_dataset_fingerprint(_fingerprinted_artifact_paths(artifacts))
    manifest = read_dataset_manifest(dataset_dir)
    save_dataset_manifest(dataset_dir, replace(manifest, dataset_fingerprint=fingerprint))


def _dataset_already_generated(dataset_dir: Path) -> bool:
    """Return True when dataset_dir already holds a complete, unchanged dataset.

    Requires the manifest's declared artifacts to exist and, when the manifest
    carries a fingerprint, for that fingerprint to still match the files on
    disk. A manifest without one predates fingerprinting or came from outside
    the generation pipeline: nothing is known about staleness there, so the
    original existence-only answer stands rather than forcing a regeneration.

    Args:
        dataset_dir: Candidate dataset directory.

    Returns:
        True when regeneration can be skipped.
    """
    try:
        artifacts = resolve_dataset_artifacts(dataset_dir)
        manifest = read_dataset_manifest(dataset_dir)
    except FileNotFoundError:
        return False

    paths = _fingerprinted_artifact_paths(artifacts)
    if not all(path.exists() for path in paths):
        return False
    if manifest.dataset_fingerprint is None:
        return True
    return manifest.dataset_fingerprint == compute_dataset_fingerprint(paths)


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
) -> str:
    """Build a persisted dataset by composing domain payload generation with storage.

    Skips regeneration when `dataset_dir` already holds a complete dataset
    matching the requested format whose artifacts still match the fingerprint
    recorded when they were written, unless `force=True`. Each fresh write
    stamps that fingerprint into the manifest, so a dataset whose files were
    touched or replaced out-of-band regenerates instead of being reused.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: How the dataset is assembled — strategy budgets, RNG controls,
            replacement policy and normalization.
        dataset_dir: Target directory for the persisted dataset.
        dataset_format: Storage format family for the persisted artifacts.
        storage: Optional storage override; defaults to the format's storage.
        accumulator: Optional accumulator override; defaults to the storage's.
        force: Regenerate even when a complete dataset already exists.

    Returns:
        The `dataset_dir` that now holds the dataset.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    _guard_format_conflict(dataset_path, dataset_format)
    if not force and _dataset_already_generated(dataset_path):
        logger.info(f"Dataset already exists at '{dataset_dir}'; skipping regeneration.")
        return dataset_dir
    dataset_storage = storage or make_generation_dataset_storage(dataset_format)
    acc: DatasetAccumulatorPort = accumulator or dataset_storage.make_accumulator(dataset_path)
    payload = build_dataset_payload(source, _with_default_solvers(spec), accumulator=acc)
    dataset_storage.write_dataset(dataset_path, payload)
    _stamp_dataset_fingerprint(dataset_path)
    return dataset_dir
