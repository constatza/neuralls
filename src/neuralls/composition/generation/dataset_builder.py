"""Dataset persistence composition for generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from neuralls.composition.generation.csr_streaming import write_csr_streamed
from neuralls.composition.generation.default_services import make_solver
from neuralls.composition.generation.dense_streaming import write_dense_streamed
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.storage.dataset_digest import (
    current_dataset_digest,
    dataset_content_digest,
    stat_digest,
)
from neuralls.platform.storage.dense_stream import STREAMED_DENSE_FORMATS
from neuralls.platform.storage.manifest_io import read_dataset_manifest, save_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.device import ResourceUsage, track_resource_usage
from neuralls.shared.types import DatasetFormat, MatrixFormat, SparsityPattern

_GENERATION_DEVICE = torch.device("cpu")

type StreamedWriter = Callable[[SourceSpec, DatasetSpec, Path, DatasetFormat], None]

_DEFAULT_SOLVER = make_solver()
_DEFAULT_SOLVER_OVERRIDES: dict[str, Any] = {
    "residuals": _DEFAULT_SOLVER,
    "gaussian_residuals": _DEFAULT_SOLVER,
}


def _refuse_non_generation_format(dataset_format: DatasetFormat) -> None:
    """Reject formats that are readable but not generation outputs (npy)."""
    if dataset_format in STREAMED_DENSE_FORMATS:
        return
    supported = ", ".join(sorted(STREAMED_DENSE_FORMATS))
    raise ValueError(
        f"Dataset format '{dataset_format}' cannot be generated. "
        f"Supported generation formats: {supported}."
    )


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


def _stamp_dataset_identity(
    dataset_dir: Path, identity: StageIdentity | None, *, generation_usage: ResourceUsage | None
) -> None:
    """Record digests (and identity, when known) of the freshly written dataset.

    Runs after the storage backend has written both the arrays and the
    manifest. Until this stamp lands the manifest has no ``identity_key``, so a
    crash in between leaves a dataset that is regenerated, never trusted.

    Args:
        dataset_dir: Directory holding the dataset just written.
        identity: Generation identity to persist, or None for identity-less builds.
        generation_usage: Wall time/peak memory of the generation that just
            produced this dataset — stamped so a later consumer that skips
            regeneration can still charge this cost (see
            ``DatasetManifest.generation_duration_seconds``).
    """
    manifest = read_dataset_manifest(dataset_dir)
    stamped = replace(
        manifest,
        content_digest=dataset_content_digest(dataset_dir),
        stat_digest=stat_digest(dataset_dir),
        identity_key=identity.key if identity is not None else None,
        identity_components=dict(identity.components) if identity is not None else None,
        generation_duration_seconds=(
            generation_usage.wall_time_seconds if generation_usage is not None else None
        ),
        generation_peak_memory_bytes=(
            generation_usage.peak_memory_bytes if generation_usage is not None else None
        ),
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


def _streamed_writer_for(
    matrix_format: MatrixFormat, sparsity_pattern: SparsityPattern
) -> StreamedWriter:
    """The streamed writer for a matrix format.

    The sparsity pattern only applies to CSR; the dense writer takes no layout argument.
    """
    match matrix_format:
        case MatrixFormat.DENSE:
            return write_dense_streamed
        case MatrixFormat.CSR:
            return partial(write_csr_streamed, sparsity_pattern=sparsity_pattern)


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
    force: bool = False,
    identity: StageIdentity | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
    sparsity_pattern: SparsityPattern = SparsityPattern.RAGGED,
) -> str:
    """Build a persisted dataset by streaming batches from the generator to storage.

    Skips regeneration when `dataset_dir` holds a dataset stamped with the same
    generation `identity` whose artifact content still matches its stamped
    digest, unless `force=True`. Without an `identity` nothing can be proven
    about an existing dataset, so it is always regenerated. Each fresh write
    stamps the identity and content/stat digests into the manifest after the
    commit rename.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: How the dataset is assembled — strategy budgets, RNG controls,
            replacement policy and normalization.
        dataset_dir: Target directory for the persisted dataset.
        dataset_format: Storage format family; zarr or hdf5 (npy is refused).
        force: Regenerate even when a complete dataset already exists.
        identity: Generation identity used for reuse and stamped into the manifest.
        matrix_format: Storage format of the system matrices (``[output].matrix_format``).
        sparsity_pattern: Stored CSR layout (``[output].sparsity_pattern``); CSR only.

    Returns:
        The `dataset_dir` that now holds the dataset.

    Raises:
        ValueError: If `dataset_format` is not a generation format, if an existing
            dataset was written in another format, or if a strategy count is open-ended.
    """
    _refuse_non_generation_format(dataset_format)
    dataset_path = Path(dataset_dir)
    _guard_format_conflict(dataset_path, dataset_format)
    if not force and identity is not None and is_dataset_reusable(dataset_path, identity):
        logger.info(f"Dataset already exists at '{dataset_dir}'; skipping regeneration.")
        return dataset_dir
    streamed_writer = _streamed_writer_for(matrix_format, sparsity_pattern)
    # Generation stays on the CPU: numpy/scipy sampling, and the residual PCG
    # runs on CPU tensors (composition/solvers/torchalg_runner.py), so the
    # measurement must not report CUDA memory for work that never reaches it.
    with track_resource_usage(_GENERATION_DEVICE) as usage:
        streamed_writer(source, _with_default_solvers(spec), dataset_path, dataset_format)
    # Stamped only after the commit rename: an interrupted run never gets an identity.
    _stamp_dataset_identity(dataset_path, identity, generation_usage=usage())
    return dataset_dir
