"""Populations, the failure rule and per-method summaries over the observation view."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from neuralls.domain.benchmarking.uncertainty import ClusterStatistic, matrix_cluster_interval
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import (
    OUTCOME_COUNTS_SCHEMA,
    RESIDUAL_CONSISTENCY_SCHEMA,
    SUMMARY_SCHEMA,
    TIMING_NOISE_SCHEMA,
    Population,
    outcome_counts,
    residual_consistency,
    select,
    summarize,
    timing_noise,
)
from neuralls.shared.constants import DEFAULT_RANDOM_SEED, RESIDUAL_MISMATCH_FACTOR
from neuralls.shared.types import MatrixId
from tests.observation_support import AMG, BROKEN, DATASET, JACOBI

SYSTEM_METRICS = (
    MetricId.ITERATIONS,
    MetricId.SOLVE_S,
    MetricId.ONLINE_S,
    MetricId.STANDALONE_S,
)
CONVERGED_AMG_SYSTEMS = ["m1-0", "m1-1", "m1-2", "m2-0"]
STATISTICS = [
    "mean_over_systems",
    "mean_over_matrices",
    "mean_over_systems_ci_low",
    "mean_over_systems_ci_high",
    "mean_over_matrices_ci_low",
    "mean_over_matrices_ci_high",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "min",
    "max",
]
QUANTILE_LEVELS = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}


def _row(summary: pl.DataFrame, method: str) -> dict[str, object]:
    return summary.filter(pl.col("method") == method).row(0, named=True)


def _systems(frame: pl.DataFrame, method: str) -> list[str]:
    return frame.filter(pl.col("method") == method)["system_id"].to_list()


def _matrices(frame: pl.DataFrame, method: str) -> list[int]:
    return frame.filter(pl.col("method") == method)["matrix_id"].to_list()


@pytest.mark.parametrize("metric", SYSTEM_METRICS)
def test_cost_metrics_keep_converged_systems_only(
    unbalanced_observations: pl.DataFrame, metric: MetricId
) -> None:
    own = select(unbalanced_observations, metric, Population.OWN)

    assert _systems(own, AMG) == CONVERGED_AMG_SYSTEMS
    assert len(_systems(own, JACOBI)) == 9
    assert _systems(own, BROKEN) == []


def test_time_per_iteration_keeps_unconverged_but_not_solver_errors(
    unbalanced_observations: pl.DataFrame,
) -> None:
    own = select(unbalanced_observations, MetricId.TIME_PER_ITERATION_S, Population.OWN)

    assert _systems(own, AMG) == [*CONVERGED_AMG_SYSTEMS, "m2-1"]


def test_setup_time_has_one_row_per_matrix_with_a_completed_setup(
    unbalanced_observations: pl.DataFrame,
) -> None:
    own = select(unbalanced_observations, MetricId.SETUP_S, Population.OWN)

    assert _matrices(own, AMG) == [1, 2]
    assert _matrices(own, JACOBI) == [1, 2, 3, 4]
    assert _matrices(own, BROKEN) == []


def test_the_common_population_drops_systems_any_method_could_not_define(
    two_method_observations: pl.DataFrame,
) -> None:
    common = select(two_method_observations, MetricId.ITERATIONS, Population.COMMON)

    assert _systems(common, AMG) == CONVERGED_AMG_SYSTEMS
    assert _systems(common, JACOBI) == CONVERGED_AMG_SYSTEMS


def test_the_common_population_follows_each_metrics_own_failure_rule(
    two_method_observations: pl.DataFrame,
) -> None:
    per_iteration = select(
        two_method_observations, MetricId.TIME_PER_ITERATION_S, Population.COMMON
    )
    setup = select(two_method_observations, MetricId.SETUP_S, Population.COMMON)

    assert _systems(per_iteration, JACOBI) == [*CONVERGED_AMG_SYSTEMS, "m2-1"]
    assert _matrices(setup, JACOBI) == [1, 2]


def test_a_method_without_defined_units_empties_the_common_population(
    unbalanced_observations: pl.DataFrame,
) -> None:
    assert select(unbalanced_observations, MetricId.ITERATIONS, Population.COMMON).height == 0


def test_summary_has_the_declared_schema_and_one_row_per_method(
    unbalanced_observations: pl.DataFrame,
) -> None:
    summary = summarize(unbalanced_observations, MetricId.ITERATIONS, Population.OWN)

    assert dict(summary.schema) == SUMMARY_SCHEMA
    assert summary["method"].to_list() == [AMG, BROKEN, JACOBI]


def test_summary_of_the_unbalanced_method_matches_the_hand_computed_row(
    unbalanced_observations: pl.DataFrame,
) -> None:
    summary = summarize(unbalanced_observations, MetricId.ITERATIONS, Population.OWN)
    systems_ci = matrix_cluster_interval(
        {(DATASET, MatrixId(1)): [2.0, 5.0, 3.0], (DATASET, MatrixId(2)): [8.0]},
        ClusterStatistic.MEAN_OVER_SYSTEMS,
        seed=DEFAULT_RANDOM_SEED,
    )
    matrices_ci = matrix_cluster_interval(
        {(DATASET, MatrixId(1)): [2.0, 5.0, 3.0], (DATASET, MatrixId(2)): [8.0]},
        ClusterStatistic.MEAN_OVER_MATRICES,
        seed=DEFAULT_RANDOM_SEED,
    )
    assert systems_ci is not None
    assert matrices_ci is not None

    assert _row(summary, AMG) == pytest.approx(
        {
            "method": AMG,
            "metric": "iterations",
            "population": "own",
            "n_planned": 9,
            "n_defined": 4,
            "n_matrices": 2,
            "n_unconverged": 1,
            "n_solver_error": 1,
            "n_setup_failed": 2,
            "n_not_attempted": 1,
            # values 2, 5, 3 on matrix 1 and 8 on matrix 2
            "mean_over_systems": 18 / 4,
            "mean_over_matrices": (10 / 3 + 8) / 2,
            "mean_over_systems_ci_low": systems_ci.low,
            "mean_over_systems_ci_high": systems_ci.high,
            "mean_over_matrices_ci_low": matrices_ci.low,
            "mean_over_matrices_ci_high": matrices_ci.high,
            # sorted values 2, 3, 5, 8, position (n - 1) * level
            "q05": 2.15,
            "q25": 2.75,
            "q50": 4.0,
            "q75": 5.75,
            "q95": 7.55,
            "min": 2.0,
            "max": 8.0,
        }
    )


def test_the_two_means_differ_on_unbalanced_matrices(
    unbalanced_observations: pl.DataFrame,
) -> None:
    jacobi = _row(summarize(unbalanced_observations, MetricId.ITERATIONS, Population.OWN), JACOBI)

    # matrix means 12, 18, 5, 8
    assert jacobi["mean_over_systems"] == pytest.approx(108 / 9)
    assert jacobi["mean_over_matrices"] == pytest.approx(43 / 4)


def test_quantiles_equal_numpy_linear_interpolation(
    unbalanced_observations: pl.DataFrame,
) -> None:
    jacobi = _row(summarize(unbalanced_observations, MetricId.ITERATIONS, Population.OWN), JACOBI)
    values = [10, 12, 14, 16, 18, 20, 4, 6, 8]

    for column, level in QUANTILE_LEVELS.items():
        assert jacobi[column] == pytest.approx(np.quantile(values, level, method="linear"))


def test_a_method_with_no_defined_unit_has_null_statistics_and_zero_defined(
    unbalanced_observations: pl.DataFrame,
) -> None:
    broken = _row(summarize(unbalanced_observations, MetricId.ITERATIONS, Population.OWN), BROKEN)

    assert (broken["n_planned"], broken["n_defined"], broken["n_matrices"]) == (9, 0, 0)
    assert (broken["n_setup_failed"], broken["n_not_attempted"]) == (8, 1)
    assert {column: broken[column] for column in STATISTICS} == dict.fromkeys(STATISTICS)


def test_matrix_metrics_count_matrices_but_the_same_system_outcomes(
    unbalanced_observations: pl.DataFrame,
) -> None:
    summary = summarize(unbalanced_observations, MetricId.SETUP_S, Population.OWN)
    amg = _row(summary, AMG)

    assert (amg["n_planned"], amg["n_defined"], amg["n_matrices"]) == (4, 2, 2)
    assert (amg["n_unconverged"], amg["n_solver_error"]) == (1, 1)
    assert (amg["n_setup_failed"], amg["n_not_attempted"]) == (2, 1)
    assert amg["mean_over_systems"] == pytest.approx(9.0)
    assert amg["mean_over_matrices"] == pytest.approx(9.0)
    assert (amg["min"], amg["max"]) == (6.0, 12.0)


def test_common_population_summary_counts_the_dropped_systems(
    two_method_observations: pl.DataFrame,
) -> None:
    jacobi = _row(
        summarize(two_method_observations, MetricId.ITERATIONS, Population.COMMON), JACOBI
    )

    assert (jacobi["population"], jacobi["n_planned"], jacobi["n_defined"]) == ("common", 9, 4)
    # matrix 1 values 10, 12, 14 and matrix 2 value 16
    assert jacobi["mean_over_systems"] == pytest.approx(13.0)
    assert jacobi["mean_over_matrices"] == pytest.approx(14.0)


def test_the_interval_is_deterministic_for_a_seed(
    unbalanced_observations: pl.DataFrame,
) -> None:
    first = summarize(unbalanced_observations, MetricId.SOLVE_S, Population.OWN, seed=7)
    second = summarize(unbalanced_observations, MetricId.SOLVE_S, Population.OWN, seed=7)

    assert first.equals(second)


def test_the_interval_is_null_below_two_matrices(
    unbalanced_observations: pl.DataFrame,
) -> None:
    one_matrix = unbalanced_observations.filter(pl.col("matrix_id") == 1)
    amg = _row(summarize(one_matrix, MetricId.ITERATIONS, Population.OWN), AMG)

    assert amg["mean_over_systems"] == pytest.approx(10 / 3)
    assert amg["n_matrices"] == 1
    assert [amg[column] for column in STATISTICS[2:6]] == [None] * 4


def test_every_metric_summarizes_with_the_declared_schema(
    unbalanced_observations: pl.DataFrame,
) -> None:
    for metric in METRICS:
        for population in Population:
            assert (
                dict(summarize(unbalanced_observations, metric, population).schema)
                == SUMMARY_SCHEMA
            )


def test_residual_consistency_counts_clean_converged_rows(
    two_method_observations: pl.DataFrame,
) -> None:
    result = residual_consistency(two_method_observations)

    assert dict(result.schema) == RESIDUAL_CONSISTENCY_SCHEMA
    assert result.rows(named=True) == [
        {"method": AMG, "n_checked": 4, "n_above_threshold": 0, "worst_ratio": 0.5},
        {"method": JACOBI, "n_checked": 9, "n_above_threshold": 0, "worst_ratio": 0.5},
    ]


def test_residual_consistency_flags_a_true_residual_far_above_the_threshold(
    drifted_observations: pl.DataFrame,
) -> None:
    amg = (
        residual_consistency(drifted_observations)
        .filter(pl.col("method") == AMG)
        .row(0, named=True)
    )

    assert RESIDUAL_MISMATCH_FACTOR < 20
    assert amg["n_above_threshold"] == 1
    assert amg["worst_ratio"] == pytest.approx(20.0)


def test_residual_consistency_reports_a_method_with_nothing_to_check(
    unbalanced_observations: pl.DataFrame,
) -> None:
    broken = (
        residual_consistency(unbalanced_observations)
        .filter(pl.col("method") == BROKEN)
        .row(0, named=True)
    )

    assert broken == {"method": BROKEN, "n_checked": 0, "n_above_threshold": 0, "worst_ratio": None}


def test_outcome_counts_tally_every_planned_system_of_each_method_by_outcome(
    unbalanced_observations: pl.DataFrame,
) -> None:
    result = outcome_counts(unbalanced_observations)

    assert dict(result.schema) == OUTCOME_COUNTS_SCHEMA
    assert result.rows(named=True) == [
        {
            "method": AMG,
            "converged": 4,
            "unconverged": 1,
            "solver_error": 1,
            "setup_failed": 2,
            "not_attempted": 1,
        },
        {
            "method": BROKEN,
            "converged": 0,
            "unconverged": 0,
            "solver_error": 0,
            "setup_failed": 8,
            "not_attempted": 1,
        },
        {
            "method": JACOBI,
            "converged": 9,
            "unconverged": 0,
            "solver_error": 0,
            "setup_failed": 0,
            "not_attempted": 0,
        },
    ]


def test_timing_noise_is_the_median_spread_over_systems_with_a_solve_row(
    unbalanced_observations: pl.DataFrame,
) -> None:
    result = timing_noise(unbalanced_observations)

    assert dict(result.schema) == TIMING_NOISE_SCHEMA
    assert result.rows(named=True) == [
        {"method": AMG, "median_timing_spread": 0.0, "n_systems": 6},
        {"method": BROKEN, "median_timing_spread": None, "n_systems": 0},
        {"method": JACOBI, "median_timing_spread": 0.0, "n_systems": 9},
    ]
