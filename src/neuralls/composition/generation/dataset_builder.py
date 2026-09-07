"""Dataset persistence composition for generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation.data_types import NormalizeType
from neuralls.domain.generation.orchestration import build_dataset_payload
from neuralls.domain.generation.ports import DatasetAccumulatorPort
from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.storage.datasets import (
    GenerationDatasetStorage,
    make_generation_dataset_storage,
)
from neuralls.platform.storage.manifest_io import read_dataset_manifest
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


def _dataset_already_generated(dataset_dir: Path) -> bool:
    """Return True when dataset_dir already holds a complete, readable dataset."""
    try:
        artifacts = resolve_dataset_artifacts(dataset_dir)
    except FileNotFoundError:
        return False
    return all(
        path.exists()
        for path in (artifacts.matrix.path, artifacts.rhs.path, artifacts.solutions.path)
    )


def build_dataset(
    matrix_path: str,
    dataset_dir: str,
    *,
    counts: dict[str, int] | None = None,
    mix: dict[str, float] | None = None,
    total: int | None = None,
    rhs_path: str | None = None,
    solution_path: str | None = None,
    parameters_paths: tuple[str, ...] = (),
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    include_indices: tuple[int, ...] | None = None,
    exclude_indices: tuple[int, ...] = (),
    replacement: bool = False,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, Any] | None = None,
    dataset_format: DatasetFormat = "hdf5",
    storage: GenerationDatasetStorage | None = None,
    accumulator: DatasetAccumulatorPort | None = None,
    force: bool = False,
) -> str:
    """Build a persisted dataset by composing domain payload generation with storage.

    Skips regeneration when `dataset_dir` already holds a complete dataset
    matching the requested format, unless `force=True`.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    _guard_format_conflict(dataset_path, dataset_format)
    if not force and _dataset_already_generated(dataset_path):
        logger.info(f"Dataset already exists at '{dataset_dir}'; skipping regeneration.")
        return dataset_dir
    dataset_storage = storage or make_generation_dataset_storage(dataset_format)
    acc: DatasetAccumulatorPort = accumulator or dataset_storage.make_accumulator(dataset_path)
    payload = build_dataset_payload(
        matrix_path=matrix_path,
        counts=counts,
        mix=mix,
        total=total,
        rhs_path=rhs_path,
        solution_path=solution_path,
        parameters_paths=parameters_paths,
        sample_id_regex=sample_id_regex,
        enumerate_by=enumerate_by,
        include_indices=include_indices,
        exclude_indices=exclude_indices,
        replacement=replacement,
        normalize=normalize,
        matrix_norm_type=matrix_norm_type,
        shuffle=shuffle,
        seed=seed,
        strategy_overrides=strategy_overrides,
        solver_overrides={**_DEFAULT_SOLVER_OVERRIDES, **(solver_overrides or {})},
        accumulator=acc,
    )
    dataset_storage.write_dataset(dataset_path, payload)
    return dataset_dir
