"""Failure-aware profiles over explicit solve and amortized-online costs."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.profiles import ProfileSeries, ecdf_series, performance_series
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import PreconditionerName
from tests.observation_support import solved_row, synthetic_observations, unconverged_row

AMG = PreconditionerName("amg")
JACOBI = PreconditionerName("jacobi")
A = PreconditionerName("a")
B = PreconditionerName("b")


@pytest.fixture
def observations() -> pl.DataFrame:
    """Four systems; ``amg`` solves all, ``jacobi`` solves two and fails two."""
    return synthetic_observations(
        [
            solved_row(AMG, "s0", 10, 1.0, 5, online_seconds=1.5),
            solved_row(AMG, "s1", 10, 2.0, 6, online_seconds=2.5),
            solved_row(AMG, "s2", 20, 4.0, 7, online_seconds=4.5),
            solved_row(AMG, "s3", 20, 8.0, 8, online_seconds=8.5),
            solved_row(JACOBI, "s0", 10, 2.0, 10, online_seconds=3.0),
            solved_row(JACOBI, "s1", 10, 2.0, 12, online_seconds=3.0),
            unconverged_row(JACOBI, "s2", 20),
            unconverged_row(JACOBI, "s3", 20),
        ]
    )


@pytest.fixture(params=(MetricId.SOLVE_S, MetricId.ONLINE_S))
def time_metric(request: pytest.FixtureRequest) -> MetricId:
    """Every time-profile contract is exercised for both supported scenarios."""
    return request.param


def _x(series: ProfileSeries, method: str) -> list[float]:
    return series.frame.filter(pl.col("method") == method)["x"].to_list()


def _planned(series: ProfileSeries, method: str) -> tuple[int, int]:
    row = series.planned.filter(pl.col("method") == method).row(0, named=True)
    return row["n_planned"], row["n_solved"]


def _plateau(series: ProfileSeries, method: str) -> float:
    fractions = series.frame.filter(pl.col("method") == method)["fraction"]
    return fractions.max() if fractions.len() else 0.0


def _fraction_within(series: ProfileSeries, method: str, x: float) -> float:
    within = series.frame.filter((pl.col("method") == method) & (pl.col("x") <= x))["fraction"]
    return within.max() if within.len() else 0.0


def test_performance_profile_scores_failures_as_unsolved(
    observations: pl.DataFrame, time_metric: MetricId
) -> None:
    series = performance_series(observations, time_metric)

    assert _x(series, AMG) == [1.0, 1.0, 1.0, 1.0]
    expected = [1.0, 2.0] if time_metric is MetricId.SOLVE_S else [1.2, 2.0]
    assert _x(series, JACOBI) == pytest.approx(expected)
    assert _plateau(series, JACOBI) == pytest.approx(0.5)


def test_systems_no_method_solved_remain_in_performance_profile_denominator(
    time_metric: MetricId,
) -> None:
    observations = synthetic_observations(
        [
            solved_row(A, "s0", 1, 1.0, online_seconds=1.5),
            unconverged_row(A, "s1", 2),
            solved_row(B, "s0", 1, 2.0, online_seconds=2.5),
            unconverged_row(B, "s1", 2),
        ]
    )

    series = performance_series(observations, time_metric)

    assert _planned(series, A) == _planned(series, B) == (2, 1)
    assert _plateau(series, A) == pytest.approx(0.5)


def test_all_failed_methods_still_have_zero_profile_plateaus(time_metric: MetricId) -> None:
    observations = synthetic_observations(
        [unconverged_row(A, "s0", 1), unconverged_row(B, "s0", 1)]
    )

    series = performance_series(observations, time_metric)

    assert series.planned["method"].to_list() == [A, B]
    assert series.planned["n_planned"].to_list() == [1, 1]
    assert series.frame.height == 0


def test_ecdf_counts_failures_in_the_denominator(
    observations: pl.DataFrame, time_metric: MetricId
) -> None:
    series = ecdf_series(observations, time_metric)

    assert _planned(series, JACOBI) == (4, 2)
    assert _plateau(series, JACOBI) == pytest.approx(0.5)
    assert _plateau(series, AMG) == pytest.approx(1.0)


def test_fastest_method_is_selected_per_problem_not_globally(
    observations: pl.DataFrame, time_metric: MetricId
) -> None:
    series = performance_series(observations, time_metric)

    assert _fraction_within(series, AMG, 1.0) == pytest.approx(1.0)
    assert _fraction_within(series, JACOBI, 1.0) == pytest.approx(
        0.25 if time_metric is MetricId.SOLVE_S else 0.0
    )


def test_initial_guess_convergence_remains_in_iteration_ecdf() -> None:
    observations = synthetic_observations(
        [solved_row(A, "s0", 1, 0.0, 0, online_seconds=0.5), solved_row(A, "s1", 1, 1.0, 5)]
    )

    series = ecdf_series(observations, MetricId.ITERATIONS)

    assert _planned(series, A) == (2, 2)
    assert _fraction_within(series, A, 0.0) == pytest.approx(0.5)
