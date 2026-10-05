"""CSR-aware scaling and matrix norms must match the dense reference.

Oracle: the dense path (``ndarray`` division and ``np.linalg.norm``). A CSR
result is valid only if it equals the dense result and keeps the sparsity
pattern (``indices``/``indptr``) of its input.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.normalization import MatrixScale, matrix_norm, scale_csr_values
from neuralls.shared.types import MatrixNormType

SEED = 20260405
SIZE = 7
DENSITY_MASK_SEED = 11
NORM_TOLERANCE = 1e-12
SUPPORTED_SPARSE_NORMS = (
    MatrixNormType.SPECTRAL,
    MatrixNormType.FROBENIUS,
    MatrixNormType.ONE,
    MatrixNormType.INF,
)


@pytest.fixture
def dense_matrix() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    mask = np.random.default_rng(DENSITY_MASK_SEED).random((SIZE, SIZE)) < 0.4
    return rng.standard_normal((SIZE, SIZE)) * mask


@pytest.fixture
def csr_matrix(dense_matrix: np.ndarray) -> csr_array:
    return csr_array(dense_matrix)


@pytest.fixture
def scalar_scale() -> MatrixScale:
    return MatrixScale(spectral_radius_bound=2.5, dimension_scale=1.5)


@pytest.fixture
def row_scale() -> np.ndarray:
    return np.random.default_rng(SEED + 1).uniform(0.5, 2.0, size=SIZE)


@pytest.fixture
def col_scale() -> np.ndarray:
    return np.random.default_rng(SEED + 2).uniform(0.5, 2.0, size=SIZE)


def test_scalar_scale_matches_dense_and_keeps_pattern(
    dense_matrix: np.ndarray, csr_matrix: csr_array, scalar_scale: MatrixScale
) -> None:
    scaled = scalar_scale.scale_matrix(csr_matrix)
    expected = scalar_scale.scale_matrix(dense_matrix)

    assert isinstance(scaled, csr_array)
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(scaled.toarray(), expected, rtol=NORM_TOLERANCE)
    np.testing.assert_array_equal(scaled.indices, csr_matrix.indices)
    np.testing.assert_array_equal(scaled.indptr, csr_matrix.indptr)


def test_row_scale_matches_dense_and_keeps_pattern(
    dense_matrix: np.ndarray, csr_matrix: csr_array, row_scale: np.ndarray
) -> None:
    scaled = scale_csr_values(csr_matrix, row_scale, "row")
    expected = dense_matrix / row_scale[:, None]

    np.testing.assert_allclose(scaled.toarray(), expected, rtol=NORM_TOLERANCE)
    np.testing.assert_array_equal(scaled.indices, csr_matrix.indices)
    np.testing.assert_array_equal(scaled.indptr, csr_matrix.indptr)


def test_col_scale_matches_dense_and_keeps_pattern(
    dense_matrix: np.ndarray, csr_matrix: csr_array, col_scale: np.ndarray
) -> None:
    scaled = scale_csr_values(csr_matrix, col_scale, "col")
    expected = dense_matrix / col_scale[None, :]

    np.testing.assert_allclose(scaled.toarray(), expected, rtol=NORM_TOLERANCE)
    np.testing.assert_array_equal(scaled.indices, csr_matrix.indices)
    np.testing.assert_array_equal(scaled.indptr, csr_matrix.indptr)


def test_axis_scale_rejects_wrong_length(csr_matrix: csr_array) -> None:
    with pytest.raises(ValueError, match="Row scale must have shape"):
        scale_csr_values(csr_matrix, np.ones(SIZE + 1), "row")


@pytest.mark.parametrize("kind", SUPPORTED_SPARSE_NORMS, ids=lambda k: k.value)
def test_matrix_norm_csr_matches_dense(
    dense_matrix: np.ndarray, csr_matrix: csr_array, kind: MatrixNormType
) -> None:
    assert matrix_norm(csr_matrix, kind) == pytest.approx(
        matrix_norm(dense_matrix, kind), rel=NORM_TOLERANCE, abs=NORM_TOLERANCE
    )


def test_nuclear_norm_rejected_for_csr(csr_matrix: csr_array) -> None:
    with pytest.raises(ValueError, match="Nuclear norm"):
        matrix_norm(csr_matrix, MatrixNormType.NUCLEAR)
