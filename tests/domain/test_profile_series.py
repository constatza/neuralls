"""ECDF and performance-profile series over the observation view."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.profiles import (
    PLANNED_SCHEMA,
    PROFILE_SERIES_SCHEMA,
    ProfileSeries,
    ecdf_series,
    performance_series,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from tests.observation_support import AMG, BROKEN, JACOBI

type CurveValues = dict[str, list[float]]

ECDF_BUDGETS: tuple[tuple[MetricId, CurveValues], ...] = (
    (
        MetricId.ITERATIONS,
        {AMG: [2, 3, 5, 8], JACOBI: [4, 6, 8, 10, 12, 14, 16, 18, 20], BROKEN: []},
    ),
    (
        MetricId.SOLVE_S,
        {
            AMG: [1.0, 2.0, 3.0, 4.0],
            JACOBI: [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            BROKEN: [],
        },
    ),
    (
        # solve time plus the setup of the matrix over its planned RHS: amg m1 +6/3, m2 +12/3;
        # jacobi m1 +1/3, m2 +2/3, m3 +0.1/2, m4 +0.2/1
        MetricId.ONLINE_S,
        {
            AMG: [3.0, 4.0, 5.0, 8.0],
            JACOBI: [0.15, 0.25, 0.5, 0.5 + 1 / 3, 0.6 + 1 / 3, 0.7 + 1 / 3]
            + [0.8 + 2 / 3, 0.9 + 2 / 3, 1.0 + 2 / 3],
            BROKEN: [],
        },
    ),
)
"""The budgets at which each method solved a system; ``broken`` never solved one."""

PROFILE_RATIOS: tuple[tuple[MetricId, CurveValues], ...] = (
    (
        # amg is best on the four systems both solved; jacobi is best on the five only it solved
        # and has ratios 10/2, 12/5, 14/3, 16/8 on the others
        MetricId.ITERATIONS,
        {AMG: [1.0] * 4, JACOBI: [1.0] * 5 + [2.0, 2.4, 14 / 3, 5.0], BROKEN: []},
    ),
    (
        # jacobi is fastest everywhere; amg has ratios 1.0/0.5, 2.0/0.6, 3.0/0.7, 4.0/0.8
        MetricId.SOLVE_S,
        {AMG: [2.0, 2.0 / 0.6, 3.0 / 0.7, 5.0], JACOBI: [1.0] * 9, BROKEN: []},
    ),
    (
        # jacobi is fastest everywhere; amg has ratios 3/(0.5+1/3), 4/(0.6+1/3), 5/(0.7+1/3),
        # 8/(0.8+2/3)
        MetricId.ONLINE_S,
        {
            AMG: [3 / (0.5 + 1 / 3), 4 / (0.6 + 1 / 3), 5 / (0.7 + 1 / 3), 8 / (0.8 + 2 / 3)],
            JACOBI: [1.0] * 9,
            BROKEN: [],
        },
    ),
)
"""The factors to the best method at which each method solved a system."""

N_SOLVED = {AMG: 4, JACOBI: 9, BROKEN: 0}
N_PLANNED = 9


def _x(series: ProfileSeries, method: str) -> list[float]:
    return series.frame.filter(pl.col("method") == method)["x"].to_list()


def _fractions(series: ProfileSeries, method: str) -> list[float]:
    return series.frame.filter(pl.col("method") == method)["fraction"].to_list()


def _planned(series: ProfileSeries, method: str) -> dict[str, object]:
    return series.planned.filter(pl.col("method") == method).row(0, named=True)


def _assert_curves(series: ProfileSeries, expected: CurveValues) -> None:
    """Every method's curve rises by 1/9 per system, tied values sharing their whole tie."""
    assert sorted(series.planned["method"]) == sorted(expected)
    for method, values in expected.items():
        ascending = sorted(values)
        assert _x(series, method) == pytest.approx(ascending)
        assert _fractions(series, method) == pytest.approx(
            [sum(v <= x for v in ascending) / N_PLANNED for x in ascending]
        )
        assert _planned(series, method) == {
            "method": method,
            "n_planned": N_PLANNED,
            "n_solved": N_SOLVED[method],
        }


@pytest.mark.parametrize(("metric", "expected"), ECDF_BUDGETS)
def test_ecdf_series_lists_the_budget_of_every_solved_system(
    unbalanced_observations: pl.DataFrame, metric: MetricId, expected: CurveValues
) -> None:
    series = ecdf_series(unbalanced_observations, metric)

    _assert_curves(series, expected)
    assert series.n_excluded_zero_best == 0


@pytest.mark.parametrize(("metric", "expected"), PROFILE_RATIOS)
def test_performance_series_lists_the_factor_to_the_best_method(
    unbalanced_observations: pl.DataFrame, metric: MetricId, expected: CurveValues
) -> None:
    series = performance_series(unbalanced_observations, metric)

    _assert_curves(series, expected)
    assert series.n_excluded_zero_best == 0


def test_series_frames_have_the_declared_schemas(unbalanced_observations: pl.DataFrame) -> None:
    for series in (
        ecdf_series(unbalanced_observations, MetricId.SOLVE_S),
        performance_series(unbalanced_observations, MetricId.SOLVE_S),
    ):
        assert series.frame.schema == pl.Schema(PROFILE_SERIES_SCHEMA)
        assert series.planned.schema == pl.Schema(PLANNED_SCHEMA)


def test_a_method_without_solved_systems_keeps_its_planned_row(
    unbalanced_observations: pl.DataFrame,
) -> None:
    series = ecdf_series(unbalanced_observations, MetricId.SOLVE_S)

    assert _x(series, BROKEN) == []
    assert _planned(series, BROKEN) == {"method": BROKEN, "n_planned": 9, "n_solved": 0}


def test_the_curve_plateaus_below_one_by_the_failures_in_the_denominator(
    unbalanced_observations: pl.DataFrame,
) -> None:
    series = ecdf_series(unbalanced_observations, MetricId.ITERATIONS)

    assert _fractions(series, AMG) == pytest.approx([1 / 9, 2 / 9, 3 / 9, 4 / 9])
    assert _fractions(series, JACOBI)[-1] == 1.0


def test_tied_values_share_the_fraction_of_the_whole_tie(
    unbalanced_observations: pl.DataFrame,
) -> None:
    series = performance_series(unbalanced_observations, MetricId.SOLVE_S)

    # jacobi is fastest on all nine systems: nine ties at ratio 1.0, each tie at fraction 1
    jacobi = series.frame.filter(pl.col("method") == JACOBI)
    assert jacobi["x"].to_list() == [1.0] * 9
    assert jacobi["fraction"].to_list() == [1.0] * 9


def test_a_matrix_unit_metric_is_rejected(unbalanced_observations: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="setup_s"):
        ecdf_series(unbalanced_observations, MetricId.SETUP_S)
    with pytest.raises(ValueError, match="setup_s"):
        performance_series(unbalanced_observations, MetricId.SETUP_S)


def test_a_zero_best_system_leaves_numerator_and_denominator_of_every_method(
    zero_iteration_observations: pl.DataFrame,
) -> None:
    """``amg`` converges ``m1-0`` in 0 iterations, so no ratio to the best count exists there.

    Keeping that system in every denominator while giving it no ratio would lower every method's
    curve for a reason that is nobody's failure; the series drops it from numerator and
    denominator of every method and reports how many it dropped.
    """
    series = performance_series(zero_iteration_observations, MetricId.ITERATIONS)

    assert series.n_excluded_zero_best == 1
    assert _planned(series, AMG) == {"method": AMG, "n_planned": 8, "n_solved": 3}
    assert _planned(series, JACOBI) == {"method": JACOBI, "n_planned": 8, "n_solved": 8}
    assert _planned(series, BROKEN) == {"method": BROKEN, "n_planned": 8, "n_solved": 0}
    assert _x(series, AMG) == pytest.approx([1.0, 1.0, 1.0])
    assert _fractions(series, AMG) == pytest.approx([3 / 8] * 3)
    assert _x(series, JACOBI) == pytest.approx([1.0] * 5 + [2.0, 2.4, 14 / 3])
    assert _fractions(series, JACOBI)[-1] == 1.0


def test_a_zero_iteration_solve_stays_in_the_absolute_ecdf(
    zero_iteration_observations: pl.DataFrame,
) -> None:
    series = ecdf_series(zero_iteration_observations, MetricId.ITERATIONS)

    assert series.n_excluded_zero_best == 0
    assert _x(series, AMG)[0] == 0.0
    assert _planned(series, AMG) == {"method": AMG, "n_planned": 9, "n_solved": 4}
