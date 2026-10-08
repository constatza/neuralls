"""End-to-end: directory-per-sample MatrixMarket sources build a correct CSR dataset.

Two raw matrices, laid out as ``<root>/0/K_ff.mtx`` and ``<root>/1/K_ff.mtx``
(the new raw format), must flow through ``matrix_path`` as a glob
(``<root>/*/K_ff.mtx``), get read as CSR, and land in the built dataset as
two distinct stored matrices at the directory-derived ids — in both zarr and
hdf5 storage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import load_matrix_sparse_sample
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.shared.types import MatrixFormat

_SIZE = 4
_SEED = 20260610


def _spd(offset: float) -> sparse.csr_array:
    """A small SPD tridiagonal matrix, offset so each sample is distinguishable."""
    dense = (
        np.diag(np.full(_SIZE, 2.0 + offset))
        + np.diag(np.full(_SIZE - 1, -1.0), 1)
        + np.diag(np.full(_SIZE - 1, -1.0), -1)
    )
    return sparse.csr_array(dense)


@pytest.fixture
def directory_per_sample_mtx(tmp_path: Path) -> Path:
    """<root>/{0,1}/K_ff.mtx, two distinct SPD matrices."""
    root = tmp_path / "raw"
    root.mkdir()
    for i in (0, 1):
        case_dir = root / str(i)
        case_dir.mkdir()
        mmwrite(str(case_dir / "K_ff.mtx"), _spd(float(i)))
    return root


@pytest.mark.parametrize("dataset_format", ["zarr", "hdf5"])
def test_directory_per_sample_glob_builds_two_distinct_csr_matrices(
    directory_per_sample_mtx: Path, tmp_path: Path, dataset_format: str
) -> None:
    out_dir = tmp_path / f"dataset_{dataset_format}"
    build_dataset(
        SourceSpec(matrix_path=str(directory_per_sample_mtx / "*" / "K_ff.mtx")),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 2},
                seed=_SEED,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format=dataset_format,
        matrix_format=MatrixFormat.CSR,
    )

    first = load_matrix_sparse_sample(out_dir, 0)
    second = load_matrix_sparse_sample(out_dir, 1)
    assert isinstance(first, sparse.csr_array)
    assert isinstance(second, sparse.csr_array)
    assert (first - _spd(0.0)).nnz == 0
    assert (second - _spd(1.0)).nnz == 0

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (2, _SIZE)
    assert solutions.shape == (2, _SIZE)
