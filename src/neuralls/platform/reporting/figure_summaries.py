"""The frame each planned group figure draws and stores, by statistic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.cost_curves import break_even_curves
from neuralls.domain.benchmarking.paired import (
    paired_systems,
    reference_among_methods,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.solver.models.cost import OfflineStages
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.reporting.cohorts import CohortRows, by_cohort
from neuralls.domain.solver.reporting.matrix_breakdown import matrix_breakdown
from neuralls.domain.solver.reporting.summary import (
    RATIO_OF_SUMS_SCHEMA,
    SUMMARY_SCHEMA,
    summarize,
    time_per_iteration_summary,
)
from neuralls.platform.reporting.figure_metadata import (
    Statistic,
)
from neuralls.platform.reporting.figure_recipes import FigureVariant, group_recipes
from neuralls.platform.reporting.figure_table import build_cost_breakdown_table, build_group_table
from neuralls.shared.types import PreconditionerName


def figure_summaries(
    outcome: GroupReport, offline: Mapping[PreconditionerName, OfflineStages]
) -> Mapping[str, pl.DataFrame]:
    """The frame each figure planned for ``outcome`` draws and stores, by variant stem.

    Each frame is computed once, here, and is both what the drawer draws and the table written
    beside the figure. A figure that compares against the reference has no frame, and so is not
    drawn, when the group has no configured reference among its methods. ``offline`` holds the
    offline cost of every method of the group, which only the cost breakdown draws.
    """
    reference = reference_among_methods(outcome.observations, outcome.reference_selection)
    frames = (
        (variant.stem, _summary(variant, outcome, reference, offline))
        for recipe in group_recipes()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )
    return {stem: frame for stem, frame in frames if frame is not None}


def _summary(
    variant: FigureVariant,
    outcome: GroupReport,
    reference: PreconditionerName | None,
    offline: Mapping[PreconditionerName, OfflineStages],
) -> pl.DataFrame | None:
    observations = outcome.observations
    match variant.statistic:
        case Statistic.CONVERGED_COUNT:
            return summarize(observations, variant.metric, variant.population)
        case Statistic.WORKLOAD_TOTAL:
            return build_group_table(outcome.summaries)
        case Statistic.OFFLINE_ONLINE_TOTAL:
            return build_cost_breakdown_table(outcome.summaries, offline)
        case Statistic.ECDF:
            return ecdf_series(observations, variant.metric).frame
        case Statistic.PERFORMANCE_PROFILE:
            return performance_series(observations, variant.metric).frame
        case Statistic.PAIRED_VALUES:
            return (
                None
                if reference is None
                else paired_systems(observations, reference, variant.metric)
            )
        case Statistic.SPEED_RATIO_CDF:
            return (
                None
                if reference is None
                else speed_ratio_summary(observations, reference, variant.metric)
            )
        case Statistic.CUMULATIVE_COST:
            return None if reference is None else break_even_curves(observations, reference)
        case (
            Statistic.MEAN_OVER_SYSTEMS
            | Statistic.MEAN_OVER_MATRICES
            | Statistic.MEAN_AND_MEDIAN
            | Statistic.QUANTILES
        ):
            metric, population = variant.metric, variant.population
            return by_cohort(
                observations,
                lambda frame: summarize(frame, metric, population),
                _cohort_rows(outcome),
                schema=SUMMARY_SCHEMA,
            )
        case Statistic.RATIO_OF_SUMS:
            population = variant.population
            return by_cohort(
                observations,
                lambda frame: time_per_iteration_summary(frame, population),
                _cohort_rows(outcome),
                schema=RATIO_OF_SUMS_SCHEMA,
            )
        case Statistic.MEAN_PER_MATRIX:
            return matrix_breakdown(observations, variant.metric)
        case unreachable:
            assert_never(unreachable)


def _cohort_rows(outcome: GroupReport) -> CohortRows:
    """A mixture level keeps its pooled rows beside its recipes'; a recipe level has only one."""
    match outcome:
        case RecipeProfileReport():
            return CohortRows.RECIPES
        case WorkloadMixtureReport():
            return CohortRows.RECIPES_AND_MIXTURE
        case unreachable:
            assert_never(unreachable)
