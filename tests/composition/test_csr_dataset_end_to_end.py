"""A CSR-configured generation run stores sparse matrices that solve the stored systems.

The source is a seeded SPD matrix in MatrixMarket format, so the matrix never passes
through a dense file. The stored CSR sample must be the source matrix, and every
stored RHS must equal that matrix times its stored solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmread, mmwrite

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import load_matrix_sparse_sample
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.shared.types import MatrixFormat

_SIZE = 6
_SAMPLES = 3
_SEED = 20260610


@pytest.fixture
def sparse_spd_mtx(tmp_path: Path) -> Path:
    """A tridiagonal SPD matrix (1D Laplacian with a unit shift), written as MatrixMarket."""
    dense = (
        np.diag(np.full(_SIZE, 2.0))
        + np.diag(np.full(_SIZE - 1, -1.0), 1)
        + np.diag(np.full(_SIZE - 1, -1.0), -1)
    )
    spd = sparse.csr_array(dense)
    path = tmp_path / "A.mtx"
    mmwrite(str(path), spd)
    return path


def test_csr_generation_stores_the_source_matrix_and_consistent_systems(
    sparse_spd_mtx: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(matrix_path=str(sparse_spd_mtx)),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": _SAMPLES},
                seed=_SEED,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
        matrix_format=MatrixFormat.CSR,
    )

    stored = load_matrix_sparse_sample(out_dir, 0)
    source = _read_mtx(sparse_spd_mtx)
    assert isinstance(stored, sparse.csr_array)
    difference = stored - source
    assert difference.nnz == 0

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (_SAMPLES, _SIZE)
    np.testing.assert_allclose(rhs, solutions @ stored.toarray().T, rtol=1e-10, atol=1e-12)


def _read_mtx(path: Path) -> sparse.csr_array:
    return sparse.csr_array(mmread(str(path)))


def test_dense_and_csr_builds_generate_the_same_rows(sparse_spd_mtx: Path, tmp_path: Path) -> None:
    """The same source, seed and mixture give the same training rows in dense and CSR storage."""
    spec = DatasetSpec(
        mixture=MixtureSpec(
            counts={"gaussian_forward": _SAMPLES},
            seed=_SEED,
            shuffle=False,
        ),
        normalize="none",
    )
    source = SourceSpec(matrix_path=str(sparse_spd_mtx))
    dense_dir = tmp_path / "dense"
    csr_dir = tmp_path / "csr"
    build_dataset(
        source, spec, str(dense_dir), dataset_format="zarr", matrix_format=MatrixFormat.DENSE
    )
    build_dataset(source, spec, str(csr_dir), dataset_format="zarr", matrix_format=MatrixFormat.CSR)

    dense_rhs, dense_solutions = load_dense_training_arrays(dense_dir)
    csr_rhs, csr_solutions = load_dense_training_arrays(csr_dir)

    np.testing.assert_allclose(csr_rhs, dense_rhs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(csr_solutions, dense_solutions, rtol=1e-10, atol=1e-12)
