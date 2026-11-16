"""Comparison-time scaling of one linear system: the case's ``system_scaling`` and ``rhs_scaling``.

A stored matrix is never rescaled: its persisted scaling must equal the case's, and only a raw
external RHS is divided by that matrix's own ``alpha``. A raw matrix (no manifest) gets one
freshly computed ``alpha`` for both sides. ``rhs_scaling`` runs last.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array

from neuralls.composition.comparison._linear_system import (
    ScalingMismatchError,
    _prepare_linear_system,
    _scale_rhs,
    _scale_system,
)
from neuralls.domain.normalization import scale_factor
from neuralls.platform.storage.manifest import PersistedMatrixScaling
from neuralls.shared.system_scaling import (
    DimensionFactor,
    GershgorinScaling,
    NoScaling,
    RhsScaling,
)
from neuralls.shared.types import ComparisonRhsSourceKind


@pytest.fixture
def matrix() -> np.ndarray:
    """A symmetric positive-definite matrix whose Gershgorin scale is far from 1."""
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def rhs() -> np.ndarray:
    """An RHS whose norm is deliberately not 1, so unit scaling is visible."""
    return np.array([3.0, 4.0], dtype=np.float64)


def _stored(alpha: float) -> PersistedMatrixScaling:
    return PersistedMatrixScaling(scaling=GershgorinScaling(), alpha=alpha)


def test_prepared_csr_system_keeps_sparse_matrix(matrix: np.ndarray, rhs: np.ndarray) -> None:
    system = _prepare_linear_system(
        csr_array(matrix),
        rhs,
        NoScaling(),
        RhsScaling.NONE,
        rhs_source_kind=None,
        matrix_scaling=None,
        lhs=None,
    )

    assert system.matrix.layout is torch.sparse_csr
    torch.testing.assert_close(system.matrix.to_dense(), torch.as_tensor(matrix))


def test_persisted_scaling_mismatch_raises_with_both_scalings(
    matrix: np.ndarray, rhs: np.ndarray
) -> None:
    case_scaling = GershgorinScaling(dimension_factor=DimensionFactor.ONE)

    with pytest.raises(ScalingMismatchError) as error:
        _scale_system(
            matrix,
            rhs,
            case_scaling,
            rhs_source_kind=ComparisonRhsSourceKind.DATASET,
            matrix_scaling=_stored(2.0),
        )

    assert (error.value.dataset_scaling, error.value.case_scaling) == (
        GershgorinScaling(),
        case_scaling,
    )


@pytest.mark.parametrize("alpha", [2.0, 8.0], ids=["first-matrix", "second-matrix"])
def test_raw_rhs_is_divided_by_its_own_matrix_alpha(
    matrix: np.ndarray, rhs: np.ndarray, alpha: float
) -> None:
    scaled_matrix, scaled_rhs = _scale_system(
        matrix,
        rhs,
        GershgorinScaling(),
        rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
        matrix_scaling=_stored(alpha),
    )

    assert scaled_matrix is matrix
    np.testing.assert_array_equal(scaled_rhs, rhs / alpha)


@pytest.mark.parametrize(
    "kind",
    [
        ComparisonRhsSourceKind.DATASET,
        ComparisonRhsSourceKind.RAW_LHS,
        ComparisonRhsSourceKind.GAUSSIAN,
        ComparisonRhsSourceKind.SINUSOIDAL,
    ],
)
def test_stored_matrix_leaves_consistent_and_generated_rhs_untouched(
    matrix: np.ndarray, rhs: np.ndarray, kind: ComparisonRhsSourceKind
) -> None:
    scaled_matrix, scaled_rhs = _scale_system(
        matrix, rhs, GershgorinScaling(), rhs_source_kind=kind, matrix_scaling=_stored(2.0)
    )

    assert scaled_matrix is matrix
    assert scaled_rhs is rhs


@pytest.mark.parametrize("as_csr", [False, True], ids=["dense", "csr"])
def test_raw_matrix_and_rhs_share_one_fresh_alpha(
    matrix: np.ndarray, rhs: np.ndarray, as_csr: bool
) -> None:
    alpha = scale_factor(matrix, GershgorinScaling())
    loaded = csr_array(matrix) if as_csr else matrix

    scaled_matrix, scaled_rhs = _scale_system(
        loaded,
        rhs,
        GershgorinScaling(),
        rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
        matrix_scaling=None,
    )

    dense = scaled_matrix.toarray() if isinstance(scaled_matrix, csr_array) else scaled_matrix
    assert isinstance(scaled_matrix, csr_array) is as_csr
    np.testing.assert_array_equal(dense, matrix / alpha)
    np.testing.assert_array_equal(scaled_rhs, rhs / alpha)


def test_raw_matrix_leaves_a_generated_rhs_untouched(matrix: np.ndarray, rhs: np.ndarray) -> None:
    scaled_matrix, scaled_rhs = _scale_system(
        matrix,
        rhs,
        GershgorinScaling(),
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        matrix_scaling=None,
    )

    np.testing.assert_array_equal(scaled_matrix, matrix / scale_factor(matrix, GershgorinScaling()))
    assert scaled_rhs is rhs


def test_unit_rhs_scaling_divides_by_the_rhs_norm(rhs: np.ndarray) -> None:
    np.testing.assert_array_equal(_scale_rhs(rhs, RhsScaling.UNIT), rhs / 5.0)


def test_unit_rhs_scaling_leaves_a_zero_rhs_alone() -> None:
    zero_rhs = np.zeros(2, dtype=np.float64)

    np.testing.assert_array_equal(_scale_rhs(zero_rhs, RhsScaling.UNIT), zero_rhs)


def test_unit_rhs_scaling_runs_after_system_scaling_and_keeps_the_known_solution_check(
    matrix: np.ndarray, rhs: np.ndarray
) -> None:
    """The known ``lhs`` is checked against the system-scaled RHS, before the unit scaling."""
    lhs = np.linalg.solve(matrix, rhs)

    system = _prepare_linear_system(
        matrix,
        rhs,
        GershgorinScaling(),
        RhsScaling.UNIT,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        matrix_scaling=_stored(2.0),
        lhs=lhs,
    )

    torch.testing.assert_close(system.rhs, torch.as_tensor(rhs / 5.0))
    torch.testing.assert_close(system.matrix, torch.as_tensor(matrix))
