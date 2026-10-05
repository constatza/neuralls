"""Matrix validation for dense and CSR inputs.

The finiteness check on CSR inspects stored values only; implicit zeros are
never stored and cannot be non-finite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.solver.utils.validation import validate_matrix, validate_rhs_vector

SEED = 20260406
SIZE = 5


@pytest.fixture
def dense_matrix() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    mask = np.random.default_rng(SEED + 1).random((SIZE, SIZE)) < 0.5
    return rng.standard_normal((SIZE, SIZE)) * mask


@pytest.fixture
def csr_matrix(dense_matrix: np.ndarray) -> csr_array:
    return csr_array(dense_matrix)


@pytest.fixture
def non_square_csr() -> csr_array:
    rng = np.random.default_rng(SEED + 2)
    mask = np.random.default_rng(SEED + 3).random((SIZE, SIZE + 1)) < 0.5
    return csr_array(rng.standard_normal((SIZE, SIZE + 1)) * mask)


@pytest.fixture
def nonfinite_csr(csr_matrix: csr_array) -> csr_array:
    data = csr_matrix.data.copy()
    data[0] = np.nan
    return csr_array((data, csr_matrix.indices, csr_matrix.indptr), shape=csr_matrix.shape)


def test_valid_csr_passes(csr_matrix: csr_array) -> None:
    validate_matrix(csr_matrix)


def test_non_square_csr_raises(non_square_csr: csr_array) -> None:
    with pytest.raises(ValueError, match="Matrix must be square"):
        validate_matrix(non_square_csr)


def test_nonfinite_stored_value_in_csr_raises(nonfinite_csr: csr_array) -> None:
    with pytest.raises(ValueError, match="Matrix contains non-finite values"):
        validate_matrix(nonfinite_csr)


def test_dense_nonfinite_message_unchanged(dense_matrix: np.ndarray) -> None:
    dense_matrix[0, 0] = np.inf
    with pytest.raises(ValueError, match="^Matrix contains non-finite values$"):
        validate_matrix(dense_matrix)


def test_rhs_length_checked_against_csr_shape(csr_matrix: csr_array) -> None:
    with pytest.raises(ValueError, match="doesn't match matrix size"):
        validate_rhs_vector(np.ones(SIZE + 1), csr_matrix)
