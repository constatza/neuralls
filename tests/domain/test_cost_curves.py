"""Cumulative cost against the number of right-hand sides of one matrix, from the break-even means."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.benchmarking.cost_curves import (
    BREAK_EVEN_HORIZON_LIMIT,
    COST_CURVE_SCHEMA,
    BreakEvenState,
    CostSide,
    break_even_curves,
)
from neuralls.domain.benchmarking.paired import break_even
from tests.observation_support import AMG, BROKEN, JACOBI


def _curve(frame: pl.DataFrame, method: str, side: CostSide) -> pl.DataFrame:
    return frame.filter(pl.col("method") == method, pl.col("side") == side.value)


def test_the_curves_have_the_cost_curve_schema(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    frame = break_even_curves(setup_heavy_fast_solver_observations, JACOBI)

    assert frame.schema == pl.Schema(COST_CURVE_SCHEMA)


def test_each_side_costs_its_mean_setup_plus_n_mean_solves(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    """amg: 9 + 0.25 n; jacobi, on the systems common to the pair: 1.5 + 0.65 n."""
    frame = break_even_curves(setup_heavy_fast_solver_observations, JACOBI)

    method = _curve(frame, AMG, CostSide.METHOD)
    reference = _curve(frame, AMG, CostSide.REFERENCE)

    n = method["n_rhs"].to_list()
    assert method["cost_s"].to_list() == pytest.approx([9.0 + 0.25 * k for k in n])
    assert reference["cost_s"].to_list() == pytest.approx([1.5 + 0.65 * k for k in n])


def test_the_curves_run_from_one_rhs_past_the_break_even_count(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    """n* = 18.75 is beyond the three planned RHS per matrix, so the curve reaches 19."""
    frame = break_even_curves(setup_heavy_fast_solver_observations, JACOBI)

    assert _curve(frame, AMG, CostSide.METHOD)["n_rhs"].to_list() == list(range(1, 20))


def test_a_method_that_breaks_even_carries_its_count_and_cost_there(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    frame = break_even_curves(setup_heavy_fast_solver_observations, JACOBI)

    marks = frame.select("method", "break_even", "break_even_rhs", "break_even_cost_s").unique()

    assert marks.rows(named=True) == [
        pytest.approx(
            {
                "method": AMG,
                "break_even": BreakEvenState.BREAKS_EVEN.value,
                "break_even_rhs": 18.75,
                "break_even_cost_s": 9.0 + 0.25 * 18.75,
            }
        )
    ]


def test_the_break_even_count_is_the_one_break_even_computes(
    setup_heavy_fast_solver_observations: pl.DataFrame,
) -> None:
    frame = break_even_curves(setup_heavy_fast_solver_observations, JACOBI)

    stored = frame.select("method", "break_even_rhs").unique().sort("method")
    computed = break_even(setup_heavy_fast_solver_observations, JACOBI).select(
        "method", "break_even_rhs"
    )
    assert_frame_equal(stored, computed)


def test_a_method_not_faster_per_rhs_never_breaks_even_and_keeps_its_curves(
    two_method_observations: pl.DataFrame,
) -> None:
    """amg solves slower than jacobi, so its curve never meets jacobi's from below."""
    frame = break_even_curves(two_method_observations, JACOBI)

    assert frame.select("break_even", "break_even_rhs", "break_even_cost_s").unique().rows() == [
        (BreakEvenState.NEVER.value, None, None)
    ]
    assert _curve(frame, AMG, CostSide.METHOD)["cost_s"].null_count() == 0


def test_without_a_break_even_count_the_curves_span_the_planned_rhs_per_matrix(
    two_method_observations: pl.DataFrame,
) -> None:
    """The largest matrix plans three RHS."""
    frame = break_even_curves(two_method_observations, JACOBI)

    assert _curve(frame, AMG, CostSide.METHOD)["n_rhs"].to_list() == [1, 2, 3]


def test_a_method_never_costing_more_breaks_even_at_zero(
    two_method_observations: pl.DataFrame,
) -> None:
    """Against amg, jacobi has the cheaper setup (1.5 s) and the faster solves (0.65 s)."""
    frame = break_even_curves(two_method_observations, AMG)

    assert frame.select("break_even", "break_even_rhs", "break_even_cost_s").unique().rows() == [
        (BreakEvenState.BREAKS_EVEN.value, 0.0, pytest.approx(1.5))
    ]


def test_a_method_without_a_common_system_has_no_costs(
    unbalanced_observations: pl.DataFrame,
) -> None:
    frame = break_even_curves(unbalanced_observations, JACOBI).filter(pl.col("method") == BROKEN)

    assert frame.select("break_even", "break_even_rhs").unique().rows() == [
        (BreakEvenState.NO_COMMON_SYSTEMS.value, None)
    ]
    assert frame["cost_s"].null_count() == frame.height


def test_every_compared_method_has_a_curve_for_each_side(
    unbalanced_observations: pl.DataFrame,
) -> None:
    frame = break_even_curves(unbalanced_observations, JACOBI)

    assert frame.select("method", "side").unique().sort("method", "side").rows() == [
        (AMG, CostSide.METHOD.value),
        (AMG, CostSide.REFERENCE.value),
        (BROKEN, CostSide.METHOD.value),
        (BROKEN, CostSide.REFERENCE.value),
    ]


def test_a_distant_break_even_count_does_not_extend_the_curves_past_the_limit(
    two_method_observations: pl.DataFrame,
) -> None:
    """A per-RHS saving of a microsecond against jacobi's 0.65 s puts n* near 7.5e6."""
    nearly_as_slow = two_method_observations.with_columns(
        solve_s=pl.when(pl.col("method") == AMG)
        .then(pl.lit(0.65 - 1e-6))
        .otherwise(pl.col("solve_s"))
    )
    frame = break_even_curves(nearly_as_slow, JACOBI)

    assert frame["n_rhs"].max() == BREAK_EVEN_HORIZON_LIMIT
    assert frame.select(pl.col("break_even_rhs").max()).item() == pytest.approx(7.5e6)


def test_a_reference_absent_from_the_observations_is_rejected(
    two_method_observations: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="reference"):
        break_even_curves(two_method_observations, BROKEN)
