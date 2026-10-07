"""A plain generation build commits a complete dataset through the streamed writers."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat, MatrixFormat

_STREAMED_FORMATS: tuple[DatasetFormat, ...] = ("zarr", "hdf5")
_PLANNED_ROWS = 8


@pytest.mark.parametrize("dataset_format", _STREAMED_FORMATS)
@pytest.mark.parametrize("matrix_format", [MatrixFormat.DENSE, MatrixFormat.CSR])
def test_build_uses_only_streamed_path(
    dataset_format: DatasetFormat,
    matrix_format: MatrixFormat,
    atomic_matrix_dir: Path,
    atomic_spec: DatasetSpec,
    tmp_path: Path,
) -> None:
    source = SourceSpec(matrix_path=str(atomic_matrix_dir / "A_000.txt"))
    dataset_dir = tmp_path / "ds"

    build_dataset(
        source,
        atomic_spec,
        str(dataset_dir),
        dataset_format=dataset_format,
        matrix_format=matrix_format,
    )

    assert (dataset_dir / DATASET_MANIFEST_FILENAME).exists()


def test_streamed_build_stores_exactly_the_planned_rows(
    atomic_matrix_dir: Path,
    atomic_spec: DatasetSpec,
    tmp_path: Path,
) -> None:
    """The stored row count equals the plan total, which the streamed writer checks before commit."""
    source = SourceSpec(matrix_path=str(atomic_matrix_dir / "A_000.txt"))
    dataset_dir = tmp_path / "ds"

    build_dataset(source, atomic_spec, str(dataset_dir), dataset_format="zarr")

    assert read_dataset_manifest(dataset_dir).rhs.shape[0] == _PLANNED_ROWS
