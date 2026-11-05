"""Planning a report group yields one summary frame per figure variant, in its table's schema.

A figure draws, and its sibling table stores, the same frame, so the frame is computed once per
variant while planning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import VARIANT_METRIC
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.benchmarking.cost_curves import break_even_curves
from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    paired_systems,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.benchmarking.selection import NoReferenceConfigured
from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.cohorts import (
    COHORT_COLUMN,
    MIXTURE_COHORT,
    CohortRows,
    by_cohort,
)
from neuralls.domain.solver.reporting.matrix_breakdown import matrix_breakdown
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import (
    RATIO_OF_SUMS_SCHEMA,
    SUMMARY_SCHEMA,
    Population,
    summarize,
    time_per_iteration_summary,
)
from neuralls.platform.reporting import figure_planning
from neuralls.platform.reporting.figure_metadata import (
    GROUP_FIGURE_KINDS,
    TABLE_SCHEMAS,
    FigureKind,
    TableKind,
)
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    FigureVariant,
    GroupFigureRecipe,
    figure_summaries,
    group_recipe,
    history_variant,
)
from neuralls.platform.reporting.figure_table import build_group_table
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.types import PreconditionerName

JACOBI = PreconditionerName("jacobi")
_MEAN_STEMS = (
    "iteration_means",
    "setup_means",
    "time_means_online_common",
    "time_per_iteration",
    "iteration_distribution",
    "time_distribution_online",
    "convergence_factor",
)


def _recipe_summaries(
    report: RecipeProfileReport, metric: MetricId, population: Population = Population.OWN
) -> pl.DataFrame:
    return by_cohort(
        report.observations,
        lambda observations: summarize(observations, metric, population),
        CohortRows.RECIPES,
        schema=SUMMARY_SCHEMA,
    )


type SummaryOf = Callable[[RecipeProfileReport], pl.DataFrame]


def _variants_planned_for(
    report_type: type[RecipeProfileReport | WorkloadMixtureReport],
) -> list[tuple[FigureKind, FigureVariant]]:
    return [
        (recipe.kind, variant)
        for recipe in FIGURE_RECIPES.values()
        if isinstance(recipe, GroupFigureRecipe) and report_type in recipe.applies_to
        for variant in recipe.variants
    ]


_PROFILE_VARIANTS = _variants_planned_for(RecipeProfileReport)
_MIXTURE_VARIANTS = _variants_planned_for(WorkloadMixtureReport)
_PROFILE_STEMS = {variant.stem for _, variant in _PROFILE_VARIANTS}
_MIXTURE_STEMS = {variant.stem for _, variant in _MIXTURE_VARIANTS}

_EXPECTED_SUMMARY: dict[str, SummaryOf] = {
    "aggregate_bars": lambda report: summarize(
        report.observations, MetricId.ITERATIONS, Population.OWN
    ),
    "paired_time_scatter": lambda report: paired_systems(
        report.observations, JACOBI, REFERENCE_METRIC
    ),
    "solve_time_bars": lambda report: build_group_table(report.summaries),
    "online_time_bars": lambda report: build_group_table(report.summaries),
    "ecdf_iterations": lambda report: ecdf_series(report.observations, MetricId.ITERATIONS).frame,
    "ecdf_time_solve": lambda report: ecdf_series(report.observations, MetricId.SOLVE_S).frame,
    "ecdf_time_online": lambda report: ecdf_series(report.observations, MetricId.ONLINE_S).frame,
    "performance_profile_solve": lambda report: (
        performance_series(report.observations, MetricId.SOLVE_S).frame
    ),
    "performance_profile_online": lambda report: (
        performance_series(report.observations, MetricId.ONLINE_S).frame
    ),
    "speed_ratio_cdf": lambda report: speed_ratio_summary(
        report.observations, JACOBI, REFERENCE_METRIC
    ),
    "breakeven": lambda report: break_even_curves(report.observations, JACOBI),
    "convergence_factor": lambda report: _recipe_summaries(report, MetricId.CONVERGENCE_FACTOR),
    "iteration_means": lambda report: _recipe_summaries(report, MetricId.ITERATIONS),
    "setup_means": lambda report: _recipe_summaries(report, MetricId.SETUP_S),
    **{
        f"time_means_{name}_{population}": (
            lambda report, metric=metric, population=population: _recipe_summaries(
                report, metric, population
            )
        )
        for name, metric in (
            ("solve", MetricId.SOLVE_S),
            ("online", MetricId.ONLINE_S),
            ("standalone", MetricId.STANDALONE_S),
        )
        for population in Population
    },
    "time_per_iteration": lambda report: by_cohort(
        report.observations,
        lambda observations: time_per_iteration_summary(observations, Population.OWN),
        CohortRows.RECIPES,
        schema=RATIO_OF_SUMS_SCHEMA,
    ),
    "iteration_distribution": lambda report: _recipe_summaries(report, MetricId.ITERATIONS),
    "time_distribution_solve": lambda report: _recipe_summaries(report, MetricId.SOLVE_S),
    "time_distribution_online": lambda report: _recipe_summaries(report, MetricId.ONLINE_S),
    "matrix_heatmap_iterations": lambda report: matrix_breakdown(
        report.observations, MetricId.ITERATIONS
    ),
}
"""The frame each variant stores, from the domain statistics."""


def test_every_variant_has_its_own_file_stem() -> None:
    group = [variant.stem for _, variant in _PROFILE_VARIANTS]
    history = [history_variant(kind).stem for kind in set(FigureKind) - GROUP_FIGURE_KINDS]
    stems = [*group, *history]

    assert len(stems) == len(set(stems))


def test_planning_yields_one_figure_per_variant_of_a_profile_report(
    profile_report: RecipeProfileReport, repeated_timing_presentation: GroupPresentation
) -> None:
    plans = figure_planning.figure_plans(profile_report, repeated_timing_presentation, "key", {})

    assert [plan.stem for plan in plans] == [
        variant.figure_stem("key") for _, variant in _PROFILE_VARIANTS
    ]


def test_a_profile_report_plans_a_summary_frame_for_every_variant(
    profile_report: RecipeProfileReport,
) -> None:
    summaries = figure_summaries(profile_report)

    assert set(summaries) == _PROFILE_STEMS


def test_a_mixture_report_plans_summary_frames_only_for_the_variants_it_draws(
    mixture_report: WorkloadMixtureReport,
) -> None:
    summaries = figure_summaries(mixture_report)

    assert set(summaries) == _MIXTURE_STEMS


@pytest.mark.parametrize(
    ("kind", "variant"),
    [pytest.param(kind, variant, id=variant.stem) for kind, variant in _PROFILE_VARIANTS],
)
def test_a_variant_summary_has_the_schema_of_its_recipes_table_kind(
    profile_report: RecipeProfileReport, kind: FigureKind, variant: FigureVariant
) -> None:
    summary = figure_summaries(profile_report)[variant.stem]

    assert dict(summary.schema) == TABLE_SCHEMAS[group_recipe(kind).table]


@pytest.mark.parametrize("stem", sorted(_EXPECTED_SUMMARY))
def test_a_variant_summary_is_the_domain_statistic_its_figure_draws(
    profile_report: RecipeProfileReport, stem: str
) -> None:
    summary = figure_summaries(profile_report)[stem]

    assert_frame_equal(summary, _EXPECTED_SUMMARY[stem](profile_report))


@pytest.mark.parametrize("stem", ["solve_time_bars", "online_time_bars"])
def test_a_mixture_report_summarizes_the_workload_totals_of_its_methods(
    mixture_report: WorkloadMixtureReport, stem: str
) -> None:
    summary = figure_summaries(mixture_report)[stem]

    assert_frame_equal(summary, build_group_table(mixture_report.summaries))


def test_a_report_without_a_reference_plans_no_figure_that_compares_against_one(
    profile_report: RecipeProfileReport,
) -> None:
    unreferenced = replace(profile_report, reference_selection=NoReferenceConfigured())

    assert set(figure_summaries(unreferenced)) == _PROFILE_STEMS - {
        "paired_time_scatter",
        "speed_ratio_cdf",
        "breakeven",
    }


def test_workload_bars_write_the_workload_totals_table() -> None:
    recipes = (group_recipe(kind) for kind in (FigureKind.SOLVE_TIME, FigureKind.ONLINE_TIME))

    assert {recipe.table for recipe in recipes} == {TableKind.WORKLOAD_TOTALS}


def test_profiles_and_ecdfs_write_the_series_table() -> None:
    kinds = (FigureKind.ECDF_ITERATIONS, FigureKind.ECDF_TIME, FigureKind.PERFORMANCE_PROFILE)

    assert {group_recipe(kind).table for kind in kinds} == {TableKind.SERIES}


def test_the_speed_ratio_writes_the_speed_ratio_table() -> None:
    assert group_recipe(FigureKind.SPEED_RATIO).table is TableKind.SPEED_RATIO


def test_the_break_even_figure_writes_the_cost_curves_table() -> None:
    assert group_recipe(FigureKind.BREAKEVEN).table is TableKind.COST_CURVES


@pytest.mark.parametrize("stem", sorted(VARIANT_METRIC))
def test_a_variant_names_the_metric_it_summarizes(stem: str) -> None:
    variants = {variant.stem: variant for _, variant in _PROFILE_VARIANTS}

    assert variants[stem].metric is VARIANT_METRIC[stem]


@pytest.mark.parametrize("stem", _MEAN_STEMS)
def test_a_mixture_mean_table_holds_each_recipes_rows_and_the_pooled_mixture(
    two_recipe_mixture_report: WorkloadMixtureReport, stem: str
) -> None:
    table = figure_summaries(two_recipe_mixture_report)[stem]

    assert table[COHORT_COLUMN].unique(maintain_order=True).to_list() == [
        recipe_key(GAUSSIAN),
        recipe_key(SINE),
        MIXTURE_COHORT,
    ]


def test_the_pooled_rows_of_a_mixture_are_the_summary_of_every_recipe_together(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    table = figure_summaries(two_recipe_mixture_report)["iteration_means"]

    pooled = table.filter(pl.col(COHORT_COLUMN) == MIXTURE_COHORT).drop(COHORT_COLUMN)

    assert_frame_equal(
        pooled,
        summarize(two_recipe_mixture_report.observations, MetricId.ITERATIONS, Population.OWN),
    )


def test_a_recipe_report_has_no_mixture_rows(profile_report: RecipeProfileReport) -> None:
    table = figure_summaries(profile_report)["iteration_means"]

    assert table[COHORT_COLUMN].unique().to_list() == [recipe_key(GAUSSIAN)]


def test_mean_bars_write_the_cohort_summary_table() -> None:
    kinds = (FigureKind.ITERATION_MEANS, FigureKind.TIME_MEANS, FigureKind.SETUP_MEANS)

    assert {group_recipe(kind).table for kind in kinds} == {TableKind.COHORT_SUMMARY}


def test_distributions_write_the_cohort_summary_table_their_quantiles_are_read_from() -> None:
    kinds = (FigureKind.ITERATION_DISTRIBUTION, FigureKind.TIME_DISTRIBUTION)

    assert {group_recipe(kind).table for kind in kinds} == {TableKind.COHORT_SUMMARY}


def test_the_matrix_heatmap_writes_the_matrix_breakdown_table() -> None:
    assert group_recipe(FigureKind.MATRIX_HEATMAP).table is TableKind.MATRIX_BREAKDOWN


def test_time_per_iteration_writes_the_cohort_ratio_of_sums_table() -> None:
    assert group_recipe(FigureKind.TIME_PER_ITERATION).table is TableKind.COHORT_RATIO_OF_SUMS
