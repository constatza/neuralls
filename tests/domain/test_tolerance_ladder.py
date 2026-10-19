"""Tests for the iterations-to-tolerance ladder in ``domain.solver.cost_metrics``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neuralls.domain.solver.cost_metrics import iterations_to_tolerances, tolerance_ladder
from neuralls.domain.solver.models.cg_result import CGComparisonResult


@pytest.mark.parametrize(
    ("rtol", "expected"),
    [
        (1e-9, (1e-3, 1e-6, 1e-9)),
        (1e-8, (1e-3, 1e-6, 1e-8)),
        (1e-6, (1e-3, 1e-6)),
        (1e-3, (1e-3,)),
        (1e-2, (1e-2,)),
    ],
)
def test_ladder_steps_three_decades_and_never_passes_rtol(
    rtol: float, expected: tuple[float, ...]
) -> None:
    assert tolerance_ladder(rtol) == expected


@pytest.fixture
def result() -> CGComparisonResult:
    """Residuals after 0..4 iterations: 1, 1e-2, 1e-4, 1e-7, 1e-9."""
    history = [1.0, 1e-2, 1e-4, 1e-7, 1e-9]
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=4,
        residual=history[-1],
        residual_abs=history[-1],
        residual_history_rel=history,
        residual_history_abs=history,
        preconditioner="p",
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )


def test_iterations_are_the_first_index_at_or_below_each_rung(
    result: CGComparisonResult,
) -> None:
    assert iterations_to_tolerances(result, 1e-9) == {1e-3: 2, 1e-6: 3, 1e-9: 4}


def test_a_rung_never_reached_maps_to_none(result: CGComparisonResult) -> None:
    unconverged = replace(result, residual_history_rel=[1.0, 0.5, 0.2])

    assert iterations_to_tolerances(unconverged, 1e-6) == {1e-3: None, 1e-6: None}


@pytest.mark.parametrize("rtol", [0.0, -1e-6])
def test_non_positive_rtol_is_rejected_instead_of_looping(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol must be positive"):
        tolerance_ladder(rtol)
