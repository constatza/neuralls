"""The profile, paired and speed-ratio frames computed on observations pooled over recipes.

A mixture level draws these figures once over every system of the group. The frames are per
system or per method over planned systems, so pooling must neither merge two recipes' systems nor
drop a failure from a denominator, and the matrix clusters of the speed-ratio interval must span
the recipes that share a matrix.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.observation_support import JACOBI
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.benchmarking.paired import paired_systems, speed_ratio_summary
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.solver.models.group_outcomes import WorkloadMixtureReport
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_evidence import bootstrap_of_speed_ratios
from neuralls.shared.rhs_recipe import recipe_key

_PROFILE_METRICS = (MetricId.ITERATIONS, MetricId.SOLVE_S, MetricId.ONLINE_S)


def _by_recipe(observations: pl.DataFrame) -> list[pl.DataFrame]:
    return [
        observations.filter(pl.col("recipe") == recipe_key(recipe)) for recipe in (GAUSSIAN, SINE)
    ]


@pytest.fixture
def pooled(two_recipe_mixture_report: WorkloadMixtureReport) -> pl.DataFrame:
    """Two recipes on disjoint matrices, so each recipe's frame is a part of the pooled one."""
    return two_recipe_mixture_report.observations


@pytest.fixture
def shared_matrices(pooled: pl.DataFrame) -> pl.DataFrame:
    """The same systems with the recipes interleaved on every matrix (first RHS is ``SINE``)."""
    return pooled.with_columns(
        recipe=pl.when(pl.col("rhs_ordinal") == 0)
        .then(pl.lit(recipe_key(SINE)))
        .otherwise(pl.lit(recipe_key(GAUSSIAN)))
    )


@pytest.mark.parametrize("metric", _PROFILE_METRICS)
def test_the_pooled_ecdf_counts_every_planned_system_of_every_recipe(
    pooled: pl.DataFrame, metric: MetricId
) -> None:
    parts = [ecdf_series(part, metric) for part in _by_recipe(pooled)]

    series = ecdf_series(pooled, metric)

    for column in ("n_planned", "n_solved"):
        expected = (
            pl.concat([part.planned for part in parts]).group_by("method").agg(pl.col(column).sum())
        )
        assert_frame_equal(
            series.planned.select("method", column).sort("method"), expected.sort("method")
        )
    assert (
        series.planned["n_planned"].to_list()
        == pooled.group_by("method").len().sort("method")["len"].to_list()
    )


def test_a_failure_of_one_recipe_keeps_the_pooled_curve_below_one(pooled: pl.DataFrame) -> None:
    series = ecdf_series(pooled, MetricId.ITERATIONS)

    plateau = series.frame.group_by("method").agg(pl.col("fraction").max()).sort("method")
    unsolved = series.planned.filter(pl.col("n_solved") < pl.col("n_planned"))

    assert unsolved.height > 0
    assert plateau.join(unsolved, on="method")["fraction"].to_list() == [
        solved / planned
        for solved, planned in zip(
            unsolved.filter(pl.col("n_solved") > 0)["n_solved"],
            unsolved.filter(pl.col("n_solved") > 0)["n_planned"],
            strict=True,
        )
    ]


@pytest.mark.parametrize("metric", [MetricId.SOLVE_S, MetricId.ONLINE_S])
def test_the_pooled_performance_ratios_are_the_recipes_ratios_taken_together(
    pooled: pl.DataFrame, metric: MetricId
) -> None:
    parts = [performance_series(part, metric).frame for part in _by_recipe(pooled)]

    series = performance_series(pooled, metric)

    assert_frame_equal(
        series.frame.select("method", "x").sort("method", "x"),
        pl.concat(parts).select("method", "x").sort("method", "x"),
    )


def test_the_pooled_performance_ratio_is_taken_against_the_best_method_of_each_system(
    shared_matrices: pl.DataFrame,
) -> None:
    by_system = performance_series(shared_matrices, MetricId.SOLVE_S).frame
    by_matrix_set = performance_series(
        shared_matrices.with_columns(recipe=pl.lit(recipe_key(GAUSSIAN))), MetricId.SOLVE_S
    ).frame

    assert_frame_equal(by_system.select("method", "x"), by_matrix_set.select("method", "x"))


def test_the_pooled_pairs_are_the_recipes_pairs_taken_together(pooled: pl.DataFrame) -> None:
    parts = [paired_systems(part, JACOBI, MetricId.ONLINE_S) for part in _by_recipe(pooled)]

    assert_frame_equal(
        paired_systems(pooled, JACOBI, MetricId.ONLINE_S),
        pl.concat(parts).sort("method", "dataset", "matrix_id", "system_id"),
    )


def test_the_pooled_speed_ratio_counts_the_systems_of_every_recipe(pooled: pl.DataFrame) -> None:
    parts = pl.concat(
        [speed_ratio_summary(part, JACOBI, MetricId.ONLINE_S) for part in _by_recipe(pooled)]
    )
    columns = ["n_systems_both_solved", "n_ratio", "n_reference_only", "n_method_only"]

    summary = speed_ratio_summary(pooled, JACOBI, MetricId.ONLINE_S)

    expected = parts.group_by("method").agg(pl.col(columns).sum())
    assert_frame_equal(summary.select("method", *columns).sort("method"), expected.sort("method"))


def test_the_speed_ratio_interval_clusters_by_matrix_across_the_recipes_that_share_it(
    pooled: pl.DataFrame, shared_matrices: pl.DataFrame
) -> None:
    separate = speed_ratio_summary(pooled, JACOBI, MetricId.ONLINE_S)
    interleaved = speed_ratio_summary(shared_matrices, JACOBI, MetricId.ONLINE_S)
    per_recipe = pl.concat(
        [
            speed_ratio_summary(part, JACOBI, MetricId.ONLINE_S)
            for part in _by_recipe(shared_matrices)
        ]
    )
    pooled_matrices = max(interleaved["n_matrices"].to_list())

    assert interleaved["n_matrices"].to_list() == separate["n_matrices"].to_list()
    assert max(per_recipe.group_by("method").agg(pl.col("n_matrices").sum())["n_matrices"]) > (
        pooled_matrices
    )
    assert (
        bootstrap_of_speed_ratios(shared_matrices, JACOBI, MetricId.ONLINE_S).matrices
        == pooled_matrices
    )
