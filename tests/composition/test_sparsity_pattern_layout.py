"""The configured ``SparsityPattern`` decides the stored CSR layout, not the matrix count.

Each case builds a dataset through ``write_csr_streamed`` with an explicit pattern and reads the
layout back from the manifest. Matrix sources come from fixtures in this module, so the expected
layouts do not depend on any repo config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.csr_streaming import write_csr_streamed
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.types import DatasetFormat, LayoutType, SparsityPattern

_SEED = 7
_SIZE = 6
_MATRIX_SAMPLES = 3
_BATCH_ROWS = 4
_ROWS = 8
_MISMATCHED_INDEX = 1


@pytest.fixture(params=["zarr", "hdf5"])
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    return request.param


@pytest.fixture
def spec() -> DatasetSpec:
    return DatasetSpec(
        mixture=MixtureSpec(counts={"gaussian_forward": _ROWS}, seed=_SEED, shuffle=False),
        normalize="matrix",
        write_batch_size=_BATCH_ROWS,
    )


def _write_matrices(directory: Path, matrices: list[np.ndarray]) -> SourceSpec:
    directory.mkdir()
    for index, matrix in enumerate(matrices):
        np.savetxt(directory / f"A_{index:03d}.txt", matrix)
    return SourceSpec(matrix_path=str(directory / "A_*.txt"))


@pytest.fixture
def single_matrix_source(tmp_path: Path) -> SourceSpec:
    return _write_matrices(tmp_path / "single", [_spd_same_pattern(0)])


@pytest.fixture
def same_pattern_source(tmp_path: Path) -> SourceSpec:
    return _write_matrices(
        tmp_path / "same", [_spd_same_pattern(index) for index in range(_MATRIX_SAMPLES)]
    )


@pytest.fixture
def mismatched_source(tmp_path: Path) -> SourceSpec:
    """Matrix 0 and 2 are diagonal; matrix ``_MISMATCHED_INDEX`` adds an off-diagonal entry."""
    matrices = [_diagonal(index) for index in range(_MATRIX_SAMPLES)]
    matrices[_MISMATCHED_INDEX] = matrices[_MISMATCHED_INDEX] + _off_diagonal()
    return _write_matrices(tmp_path / "mismatched", matrices)


def _diagonal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(_SEED + seed)
    return np.diag(rng.uniform(1.0, 2.0, _SIZE))


def _off_diagonal() -> np.ndarray:
    matrix = np.zeros((_SIZE, _SIZE))
    matrix[0, 1] = matrix[1, 0] = 0.5
    return matrix


def _spd_same_pattern(seed: int) -> np.ndarray:
    """Dense-valued SPD matrix: every entry is stored, so all samples share one pattern."""
    rng = np.random.default_rng(_SEED + seed)
    base = rng.standard_normal((_SIZE, _SIZE))
    return base @ base.T + 5.0 * np.eye(_SIZE)


def _layout_of(dataset: Path) -> LayoutType:
    layout = read_dataset_manifest(dataset).matrix.layout
    assert layout is not None
    return layout


def test_ragged_with_one_matrix_stores_many_matrices(
    single_matrix_source: SourceSpec,
    spec: DatasetSpec,
    dataset_format: DatasetFormat,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "ragged_one"
    write_csr_streamed(
        single_matrix_source,
        spec,
        dataset,
        dataset_format,
        sparsity_pattern=SparsityPattern.RAGGED,
    )
    assert _layout_of(dataset) is LayoutType.MANY_MATRICES


def test_shared_with_one_matrix_stores_shared_pattern(
    single_matrix_source: SourceSpec,
    spec: DatasetSpec,
    dataset_format: DatasetFormat,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "shared_one"
    write_csr_streamed(
        single_matrix_source,
        spec,
        dataset,
        dataset_format,
        sparsity_pattern=SparsityPattern.SHARED,
    )
    assert _layout_of(dataset) is LayoutType.SHARED_PATTERN


def test_shared_with_matrices_sharing_a_pattern_stores_shared_pattern(
    same_pattern_source: SourceSpec,
    spec: DatasetSpec,
    dataset_format: DatasetFormat,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "shared_same"
    write_csr_streamed(
        same_pattern_source,
        spec,
        dataset,
        dataset_format,
        sparsity_pattern=SparsityPattern.SHARED,
    )
    assert _layout_of(dataset) is LayoutType.SHARED_PATTERN


def test_shared_with_differing_patterns_raises_naming_the_sample_and_commits_nothing(
    mismatched_source: SourceSpec, spec: DatasetSpec, dataset_format: DatasetFormat, tmp_path: Path
) -> None:
    dataset = tmp_path / "shared_mismatch"
    with pytest.raises(ValueError, match=r"stored sample \d+ .*sample 0\b"):
        write_csr_streamed(
            mismatched_source,
            spec,
            dataset,
            dataset_format,
            sparsity_pattern=SparsityPattern.SHARED,
        )
    assert not dataset.exists()
