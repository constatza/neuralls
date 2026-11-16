"""CSR-aware scaling and matrix norms must match the dense reference.

Oracle: the dense path (``ndarray`` division and ``np.linalg.norm``). A CSR
result is valid only if it equals the dense result and keeps the sparsity
pattern (``indices``/``indptr``) of its input.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.normalization import matrix_norm, scale_csr_values, scale_factor, scale_matrix
from neuralls.shared.system_scaling import DimensionFactor, GershgorinScaling, NoScaling
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


SCALAR_ALPHA = 3.75


@pytest.fixture
def axis_scale() -> np.ndarray:
    return np.random.default_rng(SEED + 1).uniform(0.5, 2.0, size=SIZE)


def test_scalar_scale_matches_dense_and_keeps_pattern(
    dense_matrix: np.ndarray, csr_matrix: csr_array
) -> None:
    scaled = scale_matrix(csr_matrix, SCALAR_ALPHA)
    expected = scale_matrix(dense_matrix, SCALAR_ALPHA)

    assert isinstance(scaled, csr_array)
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(scaled.toarray(), expected, rtol=NORM_TOLERANCE)
    np.testing.assert_array_equal(scaled.indices, csr_matrix.indices)
    np.testing.assert_array_equal(scaled.indptr, csr_matrix.indptr)


@pytest.mark.parametrize(
    ("axis", "broadcast"),
    [("row", lambda scale: scale[:, None]), ("col", lambda scale: scale[None, :])],
    ids=["row", "col"],
)
def test_axis_scale_matches_dense_and_keeps_pattern(
    dense_matrix: np.ndarray,
    csr_matrix: csr_array,
    axis_scale: np.ndarray,
    axis: str,
    broadcast: Callable[[np.ndarray], np.ndarray],
) -> None:
    scaled = scale_csr_values(csr_matrix, axis_scale, axis)
    expected = dense_matrix / broadcast(axis_scale)

    np.testing.assert_allclose(scaled.toarray(), expected, rtol=NORM_TOLERANCE)
    np.testing.assert_array_equal(scaled.indices, csr_matrix.indices)
    np.testing.assert_array_equal(scaled.indptr, csr_matrix.indptr)


def test_axis_scale_rejects_wrong_length(csr_matrix: csr_array) -> None:
    with pytest.raises(ValueError):
        scale_csr_values(csr_matrix, np.ones(SIZE + 1), "row")


@pytest.mark.parametrize("kind", SUPPORTED_SPARSE_NORMS, ids=lambda k: k.value)
def test_matrix_norm_csr_matches_dense(
    dense_matrix: np.ndarray, csr_matrix: csr_array, kind: MatrixNormType
) -> None:
    assert matrix_norm(csr_matrix, kind) == pytest.approx(
        matrix_norm(dense_matrix, kind), rel=NORM_TOLERANCE, abs=NORM_TOLERANCE
    )


def test_nuclear_norm_rejected_for_csr(csr_matrix: csr_array) -> None:
    with pytest.raises(ValueError):
        matrix_norm(csr_matrix, MatrixNormType.NUCLEAR)


def _gershgorin_bound(dense: np.ndarray) -> float:
    return max(np.abs(dense).sum(axis=0).max(), np.abs(dense).sum(axis=1).max())


@pytest.mark.parametrize("stored_as_csr", [False, True], ids=["dense", "csr"])
def test_gershgorin_sqrt_n_scale_factor_is_bound_times_sqrt_n(
    dense_matrix: np.ndarray, csr_matrix: csr_array, stored_as_csr: bool
) -> None:
    matrix = csr_matrix if stored_as_csr else dense_matrix

    alpha = scale_factor(matrix, GershgorinScaling())

    assert alpha == pytest.approx(_gershgorin_bound(dense_matrix) * np.sqrt(SIZE), rel=1e-14)


@pytest.mark.parametrize("stored_as_csr", [False, True], ids=["dense", "csr"])
def test_gershgorin_one_scale_factor_is_the_bound(
    dense_matrix: np.ndarray, csr_matrix: csr_array, stored_as_csr: bool
) -> None:
    matrix = csr_matrix if stored_as_csr else dense_matrix

    alpha = scale_factor(matrix, GershgorinScaling(dimension_factor=DimensionFactor.ONE))

    assert alpha == pytest.approx(_gershgorin_bound(dense_matrix), rel=1e-14)


def test_no_scaling_scale_factor_is_one(csr_matrix: csr_array) -> None:
    assert scale_factor(csr_matrix, NoScaling()) == 1.0
