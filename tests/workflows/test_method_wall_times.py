"""Time axis of the residual/error-vs-time plots: each method on its own clock."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.platform.reporting.plots import _method_wall_times

ITERATIONS = 4
SETUP_SECONDS = 2.0
SOLVE_SECONDS = 8.0


@pytest.fixture
def result() -> CGComparisonResult:
    """A converged run: history = initial value + one entry per iteration."""
    history = [1.0 / 10**k for k in range(ITERATIONS + 1)]
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=ITERATIONS,
        residual=history[-1],
        residual_abs=history[-1],
        residual_history_rel=history,
        residual_history_abs=history,
        preconditioner="p",
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=SETUP_SECONDS,
        solve_time_seconds=SOLVE_SECONDS,
    )


def test_curve_starts_at_setup_and_ends_after_the_solve(result: CGComparisonResult) -> None:
    times = _method_wall_times(result, len(result.residual_history_rel))

    assert times[0] == SETUP_SECONDS
    assert result.solve_time_seconds is not None
    assert times[-1] == pytest.approx(SETUP_SECONDS + result.solve_time_seconds)
    assert np.all(np.diff(times) > 0)


def test_curve_does_not_depend_on_what_ran_before_it(result: CGComparisonResult) -> None:
    """Same result, same axis — no offset from other methods in the comparison."""
    assert _method_wall_times(result, 5) == _method_wall_times(result, 5)


def test_single_point_history_sits_at_the_offset(result: CGComparisonResult) -> None:
    assert _method_wall_times(result, 1) == [SETUP_SECONDS]
