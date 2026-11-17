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
)
from neuralls.domain.solver.reporting.cohorts import by_cohort
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
from neuralls.platform.reporting.figure_recipes import (
    FigureVariant,
    drawn_reference,
    drawn_variants,
)
from neuralls.platform.reporting.figure_table import (
    build_full_cost_breakdown_table,
    build_group_table,
)
from neuralls.shared.types import PreconditionerName


def figure_summaries(
    outcome: GroupReport, offline: Mapping[PreconditionerName, OfflineStages]
) -> Mapping[str, pl.DataFrame]:
    """The frame each figure planned for ``outcome`` draws and stores, by variant stem.

    Each frame is computed once, here, and is both what the drawer draws and the table written
    beside the figure. The figures are those ``drawn_variants`` gives, the same ones planning
    plans. ``offline`` holds the offline cost of every method of the group, which only the cost
    breakdown draws.
    """
    reference = reference_among_methods(outcome.observations, outcome.reference_selection)
    return {
        variant.stem: _summary(variant, outcome, reference, offline)
        for _, variant in drawn_variants(outcome, reference)
    }


def _summary(
    variant: FigureVariant,
    outcome: GroupReport,
    reference: PreconditionerName | None,
    offline: Mapping[PreconditionerName, OfflineStages],
) -> pl.DataFrame:
    observations = outcome.observations
    match variant.statistic:
        case Statistic.WORKLOAD_SUM:
            return build_group_table(outcome.summaries)
        case Statistic.FULL_COST:
            return build_full_cost_breakdown_table(outcome.summaries, offline)
        case Statistic.ECDF:
            return ecdf_series(observations, variant.metric).frame
        case Statistic.PERFORMANCE_PROFILE:
            return performance_series(observations, variant.metric).frame
        case Statistic.PAIRED_VALUES:
            return paired_systems(observations, drawn_reference(variant, reference), variant.metric)
        case Statistic.SPEED_RATIO_CDF:
            return speed_ratio_summary(
                observations, drawn_reference(variant, reference), variant.metric
            )
        case Statistic.CUMULATIVE_COST:
            return break_even_curves(observations, drawn_reference(variant, reference))
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
                schema=SUMMARY_SCHEMA,
            )
        case Statistic.RATIO_OF_SUMS:
            population = variant.population
            return by_cohort(
                observations,
                lambda frame: time_per_iteration_summary(frame, population),
                schema=RATIO_OF_SUMS_SCHEMA,
            )
        case Statistic.MEAN_PER_MATRIX:
            return matrix_breakdown(observations, variant.metric)
        case unreachable:
            assert_never(unreachable)
