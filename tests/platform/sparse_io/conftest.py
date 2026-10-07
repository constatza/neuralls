"""Shared fixtures for sparse_io tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy import sparse as scipy_sparse
from scipy.sparse import csr_array, csr_matrix

from neuralls.platform.sparse_io.hdf5_store import HDF5_GROUP_KEY
from neuralls.platform.sparse_io.protocol import SparseLocation

SEED = 0


@pytest.fixture
def csr_with_empty_rows_and_cols() -> csr_array:
    """A 4x6 CSR matrix with an empty row (row 1) and trailing empty columns (4, 5)."""
    dense = np.zeros((4, 6), dtype=np.float64)
    dense[0, 0] = 1.5
    dense[0, 3] = -2.0
    dense[2, 1] = 3.25
    dense[3, 2] = 0.5
    return csr_array(dense)


@pytest.fixture
def csr_with_stored_zero() -> csr_array:
    """A 2x2 CSR matrix whose pattern stores an explicit zero at (0, 1)."""
    data = np.array([4.0, 0.0, 5.0], dtype=np.float64)
    indices = np.array([0, 1, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=(2, 2))


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Deterministic generator for randomized fixtures."""
    return np.random.default_rng(SEED)


@pytest.fixture
def per_sample_matrices(seeded_rng: np.random.Generator) -> list[csr_array]:
    """Three samples with different shapes and sparsity patterns (MANY_MATRICES layout)."""
    return [
        csr_array(
            cast(
                csr_matrix,
                scipy_sparse.random(3, 4, density=0.5, random_state=seeded_rng, format="csr"),
            )
        ),
        csr_array(
            cast(
                csr_matrix,
                scipy_sparse.random(5, 2, density=0.3, random_state=seeded_rng, format="csr"),
            )
        ),
        csr_array(
            cast(
                csr_matrix,
                scipy_sparse.random(4, 4, density=0.6, random_state=seeded_rng, format="csr"),
            )
        ),
    ]


@pytest.fixture
def shared_pattern_matrices(
    csr_with_empty_rows_and_cols: csr_array, seeded_rng: np.random.Generator
) -> list[csr_array]:
    """Four samples over one pattern with fresh values each (SHARED_PATTERN layout)."""
    pattern = csr_with_empty_rows_and_cols
    nnz = pattern.data.shape[0]
    return [
        csr_array(
            (seeded_rng.normal(size=nnz), pattern.indices.copy(), pattern.indptr.copy()),
            shape=pattern.shape,
        )
        for _ in range(4)
    ]


@pytest.fixture
def hdf5_location(tmp_path: Path) -> SparseLocation:
    """A dataset.h5 path under tmp_path with the default group key."""
    return SparseLocation(path=tmp_path / "dataset.h5", key=HDF5_GROUP_KEY)
