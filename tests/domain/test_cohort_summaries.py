"""Time per iteration as a ratio of sums, and summaries split by recipe with a pooled mixture."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.solver.reporting.cohorts import (
    COHORT_COLUMN,
    COHORT_RATIO_OF_SUMS_SCHEMA,
    COHORT_SUMMARY_SCHEMA,
    MIXTURE_COHORT,
    CohortRows,
    by_cohort,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import (
    RATIO_OF_SUMS_SCHEMA,
    SUMMARY_SCHEMA,
    Population,
    summarize,
    time_per_iteration_summary,
)
from neuralls.shared.rhs_recipe import recipe_key
from tests.observation_support import AMG, BROKEN, JACOBI
from tests.recipe_support import GAUSSIAN, SINE

GAUSSIAN_KEY = recipe_key(GAUSSIAN)
SINE_KEY = recipe_key(SINE)


def _row(frame: pl.DataFrame, method: str) -> dict[str, object]:
    return frame.filter(pl.col("method") == method).row(0, named=True)


def _iterations_summary(observations: pl.DataFrame) -> pl.DataFrame:
    return summarize(observations, MetricId.ITERATIONS, Population.OWN)


def test_time_per_iteration_divides_summed_solve_time_by_summed_iterations(
    unbalanced_observations: pl.DataFrame,
) -> None:
    summary = time_per_iteration_summary(unbalanced_observations, Population.OWN)

    # amg: four converged systems and one unconverged one returned an iterate; the solver error
    # on m2-2 did not, so neither its time nor its iterations enter.
    amg = _row(summary, AMG)
    assert (amg["solve_s_sum"], amg["iterations_sum"]) == (18.0, 38)
    assert amg["ratio_of_sums"] == pytest.approx(18.0 / 38)
    assert (amg["n_planned"], amg["n_defined"], amg["n_matrices"]) == (9, 5, 2)


def test_time_per_iteration_is_not_the_mean_of_per_system_rates(
    unbalanced_observations: pl.DataFrame,
) -> None:
    jacobi = _row(time_per_iteration_summary(unbalanced_observations, Population.OWN), JACOBI)
    rates = unbalanced_observations.filter(pl.col("method") == JACOBI)["time_per_iteration_s"]

    assert jacobi["ratio_of_sums"] == pytest.approx(5.1 / 108)
    assert jacobi["ratio_of_sums"] != pytest.approx(rates.mean())


def test_a_method_without_an_iterate_has_zero_defined_and_null_sums(
    unbalanced_observations: pl.DataFrame,
) -> None:
    broken = _row(time_per_iteration_summary(unbalanced_observations, Population.OWN), BROKEN)

    assert (broken["n_defined"], broken["n_matrices"]) == (0, 0)
    assert (broken["solve_s_sum"], broken["iterations_sum"], broken["ratio_of_sums"]) == (
        None,
        None,
        None,
    )


def test_the_ratio_of_sums_has_its_schema_and_one_row_per_method(
    unbalanced_observations: pl.DataFrame,
) -> None:
    summary = time_per_iteration_summary(unbalanced_observations, Population.OWN)

    assert dict(summary.schema) == RATIO_OF_SUMS_SCHEMA
    assert summary["method"].to_list() == [AMG, BROKEN, JACOBI]
    assert set(summary["metric"]) == {MetricId.TIME_PER_ITERATION_S.value}


def test_recipe_cohorts_summarize_each_recipe_on_its_own_systems(
    two_recipe_observations: pl.DataFrame,
) -> None:
    table = by_cohort(
        two_recipe_observations, _iterations_summary, CohortRows.RECIPES, schema=SUMMARY_SCHEMA
    )

    expected = pl.concat(
        [
            _iterations_summary(two_recipe_observations.filter(pl.col("recipe") == key))
            .with_columns(pl.lit(key).alias(COHORT_COLUMN))
            .select(list(COHORT_SUMMARY_SCHEMA))
            for key in (GAUSSIAN_KEY, SINE_KEY)
        ]
    )
    assert_frame_equal(table, expected)


def test_a_mixture_adds_the_pooled_rows_labelled_as_the_mixture(
    two_recipe_observations: pl.DataFrame,
) -> None:
    table = by_cohort(
        two_recipe_observations,
        _iterations_summary,
        CohortRows.RECIPES_AND_MIXTURE,
        schema=SUMMARY_SCHEMA,
    )

    assert table[COHORT_COLUMN].unique(maintain_order=True).to_list() == [
        GAUSSIAN_KEY,
        SINE_KEY,
        MIXTURE_COHORT,
    ]
    pooled = table.filter(pl.col(COHORT_COLUMN) == MIXTURE_COHORT).drop(COHORT_COLUMN)
    assert_frame_equal(pooled, _iterations_summary(two_recipe_observations))


def test_cohort_tables_put_the_cohort_first_in_front_of_the_statistics_schema() -> None:
    assert COHORT_SUMMARY_SCHEMA == {COHORT_COLUMN: pl.String, **SUMMARY_SCHEMA}
    assert COHORT_RATIO_OF_SUMS_SCHEMA == {COHORT_COLUMN: pl.String, **RATIO_OF_SUMS_SCHEMA}


def test_a_view_without_systems_has_an_empty_cohort_table(
    two_recipe_observations: pl.DataFrame,
) -> None:
    table = by_cohort(
        two_recipe_observations.clear(),
        _iterations_summary,
        CohortRows.RECIPES,
        schema=SUMMARY_SCHEMA,
    )

    assert_frame_equal(table, pl.DataFrame(schema=COHORT_SUMMARY_SCHEMA))
