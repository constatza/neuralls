"""Characterization tests for `_normalize_linear_system`'s mode dispatch.

`tests/composition/test_comparison_normalization_consistency.py` covers the
``"matrix"`` mode end-to-end through `_load_linear_system`. These pin the
remaining modes (``"none"``, ``"rhs"``, ``"both"``), the unsupported-mode
error, and the persisted-normalization guard that runs ahead of the dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.composition.comparison._linear_system import _normalize_linear_system
from neuralls.platform.storage.manifest import DatasetNormalization
from neuralls.shared.types import ComparisonRhsSourceKind


@pytest.fixture
def matrix() -> np.ndarray:
    """A symmetric positive-definite matrix whose spectral radius is far from 1."""
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def rhs() -> np.ndarray:
    """An RHS whose norm is deliberately not 1, so self-normalization is visible."""
    return np.array([3.0, 4.0], dtype=np.float64)


@pytest.fixture
def unsupported_normalization() -> DatasetNormalization:
    """Persisted metadata from a normalization scheme no longer supported."""
    return DatasetNormalization(
        type="minmax", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
    )


def test_none_mode_returns_both_sides_untouched(matrix: np.ndarray, rhs: np.ndarray) -> None:
    scaled_matrix, scaled_rhs = _normalize_linear_system(matrix, rhs, "none")

    np.testing.assert_array_equal(scaled_matrix, matrix)
    np.testing.assert_array_equal(scaled_rhs, rhs)


def test_rhs_mode_self_normalizes_the_rhs_and_leaves_the_matrix(
    matrix: np.ndarray, rhs: np.ndarray
) -> None:
    scaled_matrix, scaled_rhs = _normalize_linear_system(matrix, rhs, "rhs")

    np.testing.assert_array_equal(scaled_matrix, matrix)
    assert float(np.linalg.norm(scaled_rhs)) == pytest.approx(1.0)
    np.testing.assert_allclose(scaled_rhs, rhs / np.linalg.norm(rhs))


def test_rhs_mode_leaves_a_zero_rhs_alone(matrix: np.ndarray) -> None:
    """A zero RHS has no norm to divide by and must pass through unchanged."""
    zero_rhs = np.zeros(2, dtype=np.float64)

    _scaled_matrix, scaled_rhs = _normalize_linear_system(matrix, zero_rhs, "rhs")

    np.testing.assert_array_equal(scaled_rhs, zero_rhs)


def test_both_mode_equals_matrix_mode_followed_by_rhs_self_normalization(
    matrix: np.ndarray, rhs: np.ndarray
) -> None:
    matrix_only, rhs_after_matrix = _normalize_linear_system(matrix, rhs, "matrix")
    both_matrix, both_rhs = _normalize_linear_system(matrix, rhs, "both")

    np.testing.assert_allclose(both_matrix, matrix_only)
    np.testing.assert_allclose(both_rhs, rhs_after_matrix / np.linalg.norm(rhs_after_matrix))
    assert float(np.linalg.norm(both_rhs)) == pytest.approx(1.0)


def test_both_mode_forwards_rhs_provenance_to_the_matrix_branch(
    matrix: np.ndarray, rhs: np.ndarray
) -> None:
    """A synthetic RHS is never touched by the matrix-derived scale, only self-normalized."""
    _both_matrix, both_rhs = _normalize_linear_system(
        matrix, rhs, "both", rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN
    )

    np.testing.assert_allclose(both_rhs, rhs / np.linalg.norm(rhs))


def test_unsupported_mode_is_rejected(matrix: np.ndarray, rhs: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unsupported normalize_system value"):
        _normalize_linear_system(matrix, rhs, "spectral")


@pytest.mark.parametrize("mode", ["none", "matrix", "rhs", "both"])
def test_unsupported_persisted_normalization_is_rejected_in_every_mode(
    matrix: np.ndarray,
    rhs: np.ndarray,
    unsupported_normalization: DatasetNormalization,
    mode: str,
) -> None:
    """The persisted-metadata guard runs before mode dispatch, so it fires for all modes."""
    with pytest.raises(ValueError, match="no longer supported"):
        _normalize_linear_system(matrix, rhs, mode, matrix_normalization=unsupported_normalization)
