"""Planning a report group yields one summary frame per figure variant, in its table's schema.

A figure draws, and its sibling table stores, the same frame, so the frame is computed once per
variant while planning. The tests marked ``xfail`` pin the planning interface that does not exist
yet: ``figure_planning.figure_summaries`` (summary frame by variant stem) and
``figure_metadata.TABLE_SCHEMAS``. They are reached through module attributes so the file imports
before those exist; strict ``xfail`` fails loudly once one starts to pass, which is the cue to drop
its marker.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import VARIANT_METRIC

from neuralls.domain.benchmarking.paired import REFERENCE_METRIC, speed_ratio_summary
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting import figure_metadata, figure_planning
from neuralls.platform.reporting.figure_metadata import GROUP_FIGURE_KINDS, FigureKind
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    FigureVariant,
    GroupFigureRecipe,
    group_recipe,
    history_variant,
)
from neuralls.platform.reporting.figure_table import build_group_table
from neuralls.shared.types import PreconditionerName

JACOBI = PreconditionerName("jacobi")

type SummaryOf = Callable[[RecipeProfileReport], pl.DataFrame]

_NO_SUMMARY_PLANNING = pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="planning does not compute a summary frame per figure variant yet "
    "(figure_planning.figure_summaries is missing)",
)


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
}
"""The frame each variant with an unambiguous summary stores, from the domain statistics."""


def test_every_variant_has_its_own_file_stem() -> None:
    group = [variant.stem for _, variant in _PROFILE_VARIANTS]
    history = [history_variant(kind).stem for kind in set(FigureKind) - GROUP_FIGURE_KINDS]
    stems = [*group, *history]

    assert len(stems) == len(set(stems))


def test_planning_yields_one_figure_per_variant_of_a_profile_report(
    profile_report: RecipeProfileReport, repeated_timing_presentation: GroupPresentation
) -> None:
    plans = figure_planning.figure_plans(profile_report, repeated_timing_presentation, "key", {})

    assert [plan.filename for plan in plans] == [
        variant.filename("key") for _, variant in _PROFILE_VARIANTS
    ]


@_NO_SUMMARY_PLANNING
def test_a_profile_report_plans_a_summary_frame_for_every_variant(
    profile_report: RecipeProfileReport,
) -> None:
    summaries = figure_planning.figure_summaries(profile_report)

    assert set(summaries) == _PROFILE_STEMS


@_NO_SUMMARY_PLANNING
def test_a_mixture_report_plans_summary_frames_only_for_the_variants_it_draws(
    mixture_report: WorkloadMixtureReport,
) -> None:
    summaries = figure_planning.figure_summaries(mixture_report)

    assert set(summaries) == _MIXTURE_STEMS


@_NO_SUMMARY_PLANNING
@pytest.mark.parametrize(
    ("kind", "variant"),
    [pytest.param(kind, variant, id=variant.stem) for kind, variant in _PROFILE_VARIANTS],
)
def test_a_variant_summary_has_the_schema_of_its_recipes_table_kind(
    profile_report: RecipeProfileReport, kind: FigureKind, variant: FigureVariant
) -> None:
    summary = figure_planning.figure_summaries(profile_report)[variant.stem]

    assert dict(summary.schema) == figure_metadata.TABLE_SCHEMAS[group_recipe(kind).table]


@_NO_SUMMARY_PLANNING
@pytest.mark.parametrize("stem", sorted(_EXPECTED_SUMMARY))
def test_a_variant_summary_is_the_domain_statistic_its_figure_draws(
    profile_report: RecipeProfileReport, stem: str
) -> None:
    summary = figure_planning.figure_summaries(profile_report)[stem]

    assert_frame_equal(summary, _EXPECTED_SUMMARY[stem](profile_report))


@_NO_SUMMARY_PLANNING
@pytest.mark.parametrize("stem", ["solve_time_bars", "online_time_bars"])
def test_a_mixture_report_summarizes_the_workload_totals_of_its_methods(
    mixture_report: WorkloadMixtureReport, stem: str
) -> None:
    summary = figure_planning.figure_summaries(mixture_report)[stem]

    assert_frame_equal(summary, build_group_table(mixture_report.summaries))


def test_workload_bars_write_the_workload_totals_table() -> None:
    recipes = (group_recipe(kind) for kind in (FigureKind.SOLVE_TIME, FigureKind.ONLINE_TIME))

    assert {recipe.table for recipe in recipes} == {figure_metadata.TableKind.WORKLOAD_TOTALS}


def test_profiles_and_ecdfs_write_the_series_table() -> None:
    kinds = (FigureKind.ECDF_ITERATIONS, FigureKind.ECDF_TIME, FigureKind.PERFORMANCE_PROFILE)

    assert {group_recipe(kind).table for kind in kinds} == {figure_metadata.TableKind.SERIES}


def test_the_speed_ratio_writes_the_speed_ratio_table() -> None:
    assert group_recipe(FigureKind.SPEED_RATIO).table is figure_metadata.TableKind.SPEED_RATIO


@pytest.mark.parametrize("stem", sorted(VARIANT_METRIC))
def test_a_variant_names_the_metric_it_summarizes(stem: str) -> None:
    variants = {variant.stem: variant for _, variant in _PROFILE_VARIANTS}

    assert variants[stem].metric is VARIANT_METRIC[stem]
