"""Break-even RHS count against a reference, on the population common to the pair."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.paired import BREAK_EVEN_SCHEMA, break_even
from tests.observation_support import AMG, BROKEN, JACOBI


def _row(frame: pl.DataFrame, method: str) -> dict[str, object]:
    return frame.filter(pl.col("method") == method).row(0, named=True)


def test_break_even_is_the_setup_gap_over_the_per_rhs_saving(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    """Common systems m1-0..m1-2 and m2-0, common matrices 1 and 2.

    amg: setup (6 + 12) / 2 = 9, solve (0.1 + 0.2 + 0.3 + 0.4) / 4 = 0.25.
    jacobi: setup (1 + 2) / 2 = 1.5, solve (0.5 + 0.6 + 0.7 + 0.8) / 4 = 0.65.
    n* = (9 - 1.5) / (0.65 - 0.25) = 18.75.
    """
    frame = break_even(setup_heavy_fast_solver_observations, JACOBI)

    assert frame.schema == pl.Schema(BREAK_EVEN_SCHEMA)
    assert _row(frame, AMG) == pytest.approx(
        {
            "method": AMG,
            "setup_s_mean": 9.0,
            "reference_setup_s_mean": 1.5,
            "solve_s_mean": 0.25,
            "reference_solve_s_mean": 0.65,
            "break_even_rhs": 18.75,
        }
    )


def test_a_method_that_is_not_faster_per_rhs_never_breaks_even(
    two_method_observations: pl.DataFrame,
) -> None:
    """amg solves slower than jacobi (2.5 s against 0.65 s on average): the saving is negative."""
    row = _row(break_even(two_method_observations, JACOBI), AMG)

    assert row["solve_s_mean"] == pytest.approx(2.5)
    assert row["reference_solve_s_mean"] == pytest.approx(0.65)
    assert row["break_even_rhs"] is None


def test_a_method_never_costing_more_breaks_even_at_zero(
    two_method_observations: pl.DataFrame,
) -> None:
    """Against amg, jacobi has the cheaper setup (1.5 s against 9 s) and the faster solves."""
    row = _row(break_even(two_method_observations, AMG), JACOBI)

    assert row["setup_s_mean"] == pytest.approx(1.5)
    assert row["reference_setup_s_mean"] == pytest.approx(9.0)
    assert row["break_even_rhs"] == 0.0


def test_a_method_without_a_common_system_has_null_means_and_no_break_even(
    unbalanced_observations: pl.DataFrame,
) -> None:
    assert _row(break_even(unbalanced_observations, JACOBI), BROKEN) == {
        "method": BROKEN,
        "setup_s_mean": None,
        "reference_setup_s_mean": None,
        "solve_s_mean": None,
        "reference_solve_s_mean": None,
        "break_even_rhs": None,
    }


def test_the_population_is_common_to_the_pair_not_to_every_method(
    unbalanced_observations: pl.DataFrame,
) -> None:
    """``broken`` solves nothing, yet the amg and jacobi pair keeps its four common systems."""
    row = _row(break_even(unbalanced_observations, JACOBI), AMG)

    assert row["solve_s_mean"] == pytest.approx(2.5)
    assert row["reference_solve_s_mean"] == pytest.approx(0.65)


def test_a_reference_absent_from_the_observations_is_rejected(
    two_method_observations: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="reference"):
        break_even(two_method_observations, BROKEN)
