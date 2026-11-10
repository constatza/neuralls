"""The pre-draw plan of every group figure: stems, titles and the card metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

from neuralls.domain.benchmarking.paired import (
    reference_among_methods,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.report_levels import (
    level_name,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
)
from neuralls.platform.reporting.figure_evidence import (
    FigureInterval,
    NoInterval,
    bootstrap_of_means,
    bootstrap_of_speed_ratios,
    outcome_counts_of,
    setup_sharing_of,
    timing_evidence_of,
)
from neuralls.platform.reporting.figure_metadata import (
    TABLE_SUFFIX,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigurePlan,
    FigureTable,
    GroupAggregationRule,
    GroupFigureMetadata,
    NoTable,
    SiblingTable,
    Statistic,
    TimingPolicy,
    method_source,
)
from neuralls.platform.reporting.figure_recipes import (
    FIGURE_RECIPES,
    FigureVariant,
    GroupFigureRecipe,
    HistoryFigureRecipe,
    HistoryVariant,
    group_recipes,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName


def table_of(stem: str, kind: FigureKind) -> FigureTable:
    """The table beside the figure of file stem ``stem``: its recipe's kind, if it writes one."""
    match FIGURE_RECIPES[kind]:
        case GroupFigureRecipe(table=table_kind):
            return SiblingTable(f"{stem}{TABLE_SUFFIX}", table_kind)
        case HistoryFigureRecipe():
            return NoTable()
        case unreachable:
            assert_never(unreachable)


def plan_figure[M: FigureMetadata](
    variant: FigureVariant | HistoryVariant, metadata: M, *, slug: str
) -> FigurePlan[M]:
    """The plan of ``variant`` for the report subject named by ``slug``."""
    return FigurePlan(variant.figure_stem(slug), variant.title, metadata)


def figure_plans(
    outcome: GroupReport,
    presentation: GroupPresentation,
    file_key: str,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> tuple[FigurePlan[GroupFigureMetadata], ...]:
    """Build every figure's scientific context before drawing.

    ``stored`` holds the method run of each method ``presentation`` lists.
    """
    return tuple(
        plan_figure(
            variant,
            _figure_metadata(recipe, variant, outcome, presentation, stored),
            slug=file_key,
        )
        for recipe in group_recipes()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )


def _figure_metadata(
    recipe: GroupFigureRecipe,
    variant: FigureVariant,
    outcome: GroupReport,
    presentation: GroupPresentation,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> GroupFigureMetadata:
    dataset_ids = tuple(outcome.dataset_membership)
    recipes = outcome.recipe_membership
    weights = tuple(count / outcome.n_planned_rhs for count in outcome.recipe_counts)
    reference = outcome.reference_selection
    names = sorted(presentation.methods)
    artifacts = MethodBenchmarkArtifacts()
    methods = tuple(
        FigureMethod(name, presentation.methods[name].detailed, stored[name].config_digest)
        for name in names
    )
    sources = tuple(
        method_source(stored[name].run_id, *_source_artifacts(variant.metric, artifacts))
        for name in names
    )
    aggregation_rule = (
        GroupAggregationRule.WORKLOAD_SUM
        if isinstance(outcome, WorkloadMixtureReport)
        else GroupAggregationRule.PER_PLANNED_SYSTEM
    )
    return GroupFigureMetadata(
        kind=recipe.kind,
        metric_definition=variant.metric_definition,
        metric=variant.metric,
        population=variant.population,
        statistic=variant.statistic,
        report_level=level_name(outcome.label),
        scope=presentation.scope,
        dataset_ids=dataset_ids,
        recipes=recipes,
        n_matrices=outcome.n_matrices,
        n_planned_rhs=outcome.n_planned_rhs,
        reference_selection=reference,
        failure_treatment=recipe.failure_treatment,
        timing_policy=TimingPolicy.SETUP_PER_MATRIX_PLUS_SOLVE_PER_RHS,
        recipe_counts=outcome.recipe_counts,
        aggregation_rule=aggregation_rule,
        recipe_weights=weights,
        solver_contexts=presentation.solver_contexts,
        methods=methods,
        sources=sources,
        interval=_interval(variant, outcome),
        setup_sharing=setup_sharing_of(variant.metric, outcome.observations),
        outcome_counts=outcome_counts_of(outcome.observations),
        timing=timing_evidence_of(
            variant.metric, outcome.observations, presentation.solver_contexts
        ),
    )


def _source_artifacts(metric: MetricId, artifacts: MethodBenchmarkArtifacts) -> tuple[Path, ...]:
    """The stored artifacts a figure of ``metric`` is computed from.

    Every group figure reads the setup and solve events and the summary; the convergence factor
    also reads the residual histories, so its card cites them.
    """
    events = (artifacts.setup_events, artifacts.solve_attempts, artifacts.summary)
    match metric:
        case MetricId.CONVERGENCE_FACTOR:
            return (*events, artifacts.convergence)
        case (
            MetricId.ITERATIONS
            | MetricId.SOLVE_S
            | MetricId.ONLINE_S
            | MetricId.STANDALONE_S
            | MetricId.SETUP_S
            | MetricId.TIME_PER_ITERATION_S
        ):
            return events
        case unreachable:
            assert_never(unreachable)


def _interval(variant: FigureVariant, outcome: GroupReport) -> FigureInterval:
    """The interval a variant draws; a figure that compares to a missing reference is not drawn."""
    observations = outcome.observations
    match variant.statistic:
        case Statistic.SPEED_RATIO_CDF:
            match reference_among_methods(observations, outcome.reference_selection):
                case None:
                    return NoInterval()
                case reference:
                    return bootstrap_of_speed_ratios(observations, reference, variant.metric)
        case Statistic.MEAN_OVER_SYSTEMS | Statistic.MEAN_OVER_MATRICES | Statistic.MEAN_AND_MEDIAN:
            return bootstrap_of_means(observations, variant.metric, variant.population)
        case (
            Statistic.CONVERGED_COUNT
            | Statistic.WORKLOAD_TOTAL
            | Statistic.OFFLINE_ONLINE_TOTAL
            | Statistic.ECDF
            | Statistic.PERFORMANCE_PROFILE
            | Statistic.PAIRED_VALUES
            | Statistic.CUMULATIVE_COST
            | Statistic.RATIO_OF_SUMS
            | Statistic.QUANTILES
            | Statistic.MEAN_PER_MATRIX
        ):
            return NoInterval()
        case unreachable:
            assert_never(unreachable)
