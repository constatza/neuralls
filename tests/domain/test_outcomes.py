"""A stored solve row maps to one closed outcome, whichever way it is derived."""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from neuralls.domain.solver.reporting.observations import Outcome, solve_outcome, solve_outcome_expr

SOLVE_STATUSES = ("converged", "failed")
RESIDUAL_STATUSES = ("measured", "unmeasured")

EXPECTED = {
    ("converged", "measured"): Outcome.CONVERGED,
    ("converged", "unmeasured"): Outcome.CONVERGED,
    ("failed", "measured"): Outcome.UNCONVERGED,
    ("failed", "unmeasured"): Outcome.SOLVER_ERROR,
}


def test_outcome_values_are_the_stored_vocabulary() -> None:
    assert [outcome.value for outcome in Outcome] == [
        "converged",
        "unconverged",
        "solver_error",
        "setup_failed",
        "not_attempted",
    ]


@pytest.mark.parametrize(("solve_status", "residual_status"), list(EXPECTED))
def test_scalar_derivation_maps_each_stored_status_pair(
    solve_status: str, residual_status: str
) -> None:
    assert solve_outcome(solve_status, residual_status) == EXPECTED[(solve_status, residual_status)]


def test_expression_agrees_with_the_scalar_on_every_status_pair() -> None:
    pairs = list(itertools.product(SOLVE_STATUSES, RESIDUAL_STATUSES))
    frame = pl.DataFrame(
        {
            "solve_status": [solve for solve, _ in pairs],
            "residual_status": [residual for _, residual in pairs],
        }
    )

    derived = frame.select(solve_outcome_expr().alias("outcome"))["outcome"].to_list()

    assert derived == [solve_outcome(solve, residual) for solve, residual in pairs]
    assert derived == [EXPECTED[pair] for pair in pairs]


@pytest.mark.parametrize(
    ("solve_status", "residual_status"),
    [("running", "measured"), ("failed", "partial"), ("", ""), ("Converged", "measured")],
)
def test_unknown_status_is_rejected_by_the_scalar(solve_status: str, residual_status: str) -> None:
    with pytest.raises(ValueError):
        solve_outcome(solve_status, residual_status)


@pytest.mark.parametrize(
    ("solve_status", "residual_status"),
    [("running", "measured"), ("failed", "partial")],
)
def test_unknown_status_is_rejected_by_the_expression(
    solve_status: str, residual_status: str
) -> None:
    frame = pl.DataFrame({"solve_status": [solve_status], "residual_status": [residual_status]})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        frame.select(solve_outcome_expr())
