"""The figure registry: one recipe per kind, and the pre-draw plan of every group figure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.cost_curves import break_even_curves
from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    paired_systems,
    reference_among_methods,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.report_levels import (
    level_name,
)
from neuralls.domain.solver.reporting.summary import Population, summarize
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
)
from neuralls.platform.reporting.figure_evidence import (
    FigureInterval,
    NoInterval,
    bootstrap_of_speed_ratios,
    outcome_counts_of,
    setup_sharing_of,
    timing_evidence_of,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
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
    TableKind,
    TimingPolicy,
    method_source,
)
from neuralls.platform.reporting.figure_table import build_group_table
from neuralls.platform.reporting.group_drawers import (
    AGGREGATE_BARS_DRAWER,
    BREAK_EVEN_DRAWER,
    ECDF_ITERATIONS_DRAWER,
    ECDF_TIME_DRAWER,
    ONLINE_TIME_BARS_DRAWER,
    PAIRED_TIME_DRAWER,
    PERFORMANCE_PROFILE_DRAWER,
    SOLVE_TIME_BARS_DRAWER,
    SPEED_RATIO_DRAWER,
    GroupDrawer,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName

TABLE_SUFFIX = ".parquet"
"""A figure's table shares its stem and directory; only the suffix differs."""


def variant_filename(stem: str, slug: str) -> str:
    """The file name of the figure with file stem ``stem`` for the report subject ``slug``."""
    return f"{stem}_{slug}.png"


@dataclass(frozen=True, slots=True)
class FigureVariant:
    """One group figure a kind produces, and the statistic of a metric it draws.

    ``population`` says which systems the metric is taken over and ``statistic`` what is drawn
    of it; a kind's variants differ in ``metric``, which is what tells them apart.
    """

    stem: str
    title: str
    metric_definition: str
    metric: MetricId
    population: Population
    statistic: Statistic

    def filename(self, slug: str) -> str:
        """The file name of this figure for the report subject named by ``slug``."""
        return variant_filename(self.stem, slug)


@dataclass(frozen=True, slots=True)
class HistoryVariant:
    """The one figure a convergence-history kind produces; it summarizes no metric."""

    stem: str
    title: str
    metric_definition: str

    def filename(self, slug: str) -> str:
        """The file name of this figure for the report subject named by ``slug``."""
        return variant_filename(self.stem, slug)


class UnknownFigureVariantError(LookupError):
    """A kind was asked for a metric it has no figure for."""


class NotAGroupKindError(LookupError):
    """A group figure was asked for by a kind drawn from stored method histories."""


class NotAHistoryKindError(LookupError):
    """A history figure was asked for by a kind drawn from a group report."""


_BOTH: tuple[type[GroupReport], ...] = (RecipeProfileReport, WorkloadMixtureReport)
_PROFILE: tuple[type[GroupReport], ...] = (RecipeProfileReport,)


@dataclass(frozen=True, slots=True)
class GroupFigureRecipe:
    """Everything fixed about a group figure kind.

    ``draw`` is the drawer of every variant, ``table`` the kind of table stored beside each
    figure, ``applies_to`` the group report types the kind is planned for.
    """

    kind: FigureKind
    variants: tuple[FigureVariant, ...]
    failure_treatment: FailureTreatment
    applies_to: tuple[type[GroupReport], ...]
    draw: GroupDrawer
    table: TableKind

    def __post_init__(self) -> None:
        """Require the variants to differ in metric, which selects the variant to draw."""
        metrics = [variant.metric for variant in self.variants]
        if len(set(metrics)) != len(metrics):
            raise ValueError(f"{self.kind.value} has two variants of one metric")

    def variant(self, metric: MetricId) -> FigureVariant:
        """The variant of this kind that summarizes ``metric``."""
        for variant in self.variants:
            if variant.metric is metric:
                return variant
        raise UnknownFigureVariantError(f"{self.kind.value} has no variant for metric {metric}")


@dataclass(frozen=True, slots=True)
class HistoryFigureRecipe:
    """Everything fixed about a convergence-history kind, drawn from a stored method."""

    kind: FigureKind
    variant: HistoryVariant
    failure_treatment: FailureTreatment


type FigureRecipe = GroupFigureRecipe | HistoryFigureRecipe

FIGURE_RECIPES: Mapping[FigureKind, FigureRecipe] = {
    recipe.kind: recipe
    for recipe in (
        GroupFigureRecipe(
            FigureKind.AGGREGATE_BARS,
            (
                FigureVariant(
                    "aggregate_bars",
                    "Convergence",
                    "converged systems",
                    MetricId.ITERATIONS,
                    Population.OWN,
                    Statistic.CONVERGED_COUNT,
                ),
            ),
            FailureTreatment.CENSORED,
            _BOTH,
            AGGREGATE_BARS_DRAWER,
            TableKind.SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.SOLVE_TIME,
            (
                FigureVariant(
                    "solve_time_bars",
                    "Solve Time",
                    "solve seconds (workload total)",
                    MetricId.SOLVE_S,
                    Population.OWN,
                    Statistic.WORKLOAD_TOTAL,
                ),
            ),
            FailureTreatment.NONE,
            _BOTH,
            SOLVE_TIME_BARS_DRAWER,
            TableKind.WORKLOAD_TOTALS,
        ),
        GroupFigureRecipe(
            FigureKind.ONLINE_TIME,
            (
                FigureVariant(
                    "online_time_bars",
                    "Online Time",
                    "workload online seconds (setup + solve total)",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.WORKLOAD_TOTAL,
                ),
            ),
            FailureTreatment.NONE,
            _BOTH,
            ONLINE_TIME_BARS_DRAWER,
            TableKind.WORKLOAD_TOTALS,
        ),
        GroupFigureRecipe(
            FigureKind.PAIRED_TIME,
            (
                FigureVariant(
                    "paired_time_scatter",
                    "Online Time",
                    "amortized online seconds per paired RHS; asymmetric failures "
                    "axis-censored; both-failed pairs omitted",
                    REFERENCE_METRIC,
                    Population.OWN,
                    Statistic.PAIRED_VALUES,
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
            PAIRED_TIME_DRAWER,
            TableKind.PAIRED,
        ),
        GroupFigureRecipe(
            FigureKind.SPEED_RATIO,
            (
                FigureVariant(
                    "speed_ratio_cdf",
                    "Speed Ratio",
                    "reference/method amortized-online speed ratio",
                    REFERENCE_METRIC,
                    Population.OWN,
                    Statistic.SPEED_RATIO_CDF,
                ),
            ),
            FailureTreatment.OMITTED,
            _PROFILE,
            SPEED_RATIO_DRAWER,
            TableKind.SPEED_RATIO,
        ),
        GroupFigureRecipe(
            FigureKind.PERFORMANCE_PROFILE,
            (
                FigureVariant(
                    "performance_profile_solve",
                    "Performance Profile",
                    "solve-seconds performance ratio",
                    MetricId.SOLVE_S,
                    Population.OWN,
                    Statistic.PERFORMANCE_PROFILE,
                ),
                FigureVariant(
                    "performance_profile_online",
                    "Performance Profile",
                    "amortized-online-seconds performance ratio",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.PERFORMANCE_PROFILE,
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
            PERFORMANCE_PROFILE_DRAWER,
            TableKind.SERIES,
        ),
        GroupFigureRecipe(
            FigureKind.ECDF_ITERATIONS,
            (
                FigureVariant(
                    "ecdf_iterations",
                    "Iterations",
                    "iterations",
                    MetricId.ITERATIONS,
                    Population.OWN,
                    Statistic.ECDF,
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
            ECDF_ITERATIONS_DRAWER,
            TableKind.SERIES,
        ),
        GroupFigureRecipe(
            FigureKind.ECDF_TIME,
            (
                FigureVariant(
                    "ecdf_time_solve",
                    "Solve Time",
                    "solve seconds per planned RHS",
                    MetricId.SOLVE_S,
                    Population.OWN,
                    Statistic.ECDF,
                ),
                FigureVariant(
                    "ecdf_time_online",
                    "Online Time",
                    "amortized online seconds per planned RHS",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.ECDF,
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
            ECDF_TIME_DRAWER,
            TableKind.SERIES,
        ),
        GroupFigureRecipe(
            FigureKind.BREAKEVEN,
            (
                FigureVariant(
                    "breakeven",
                    "Break-even",
                    "mean setup + n x mean solve seconds per RHS on one matrix, means over the "
                    "matrices and systems common to the method and the reference",
                    MetricId.STANDALONE_S,
                    Population.COMMON,
                    Statistic.CUMULATIVE_COST,
                ),
            ),
            FailureTreatment.OMITTED,
            _PROFILE,
            BREAK_EVEN_DRAWER,
            TableKind.COST_CURVES,
        ),
        HistoryFigureRecipe(
            FigureKind.RESIDUAL_HISTORY,
            HistoryVariant(
                "residual_history", "Residual History", "relative residual by iteration"
            ),
            FailureTreatment.NONE,
        ),
        HistoryFigureRecipe(
            FigureKind.ENERGY_HISTORY,
            HistoryVariant(
                "energy_history", "Energy History", "relative energy error by iteration"
            ),
            FailureTreatment.NONE,
        ),
    )
}
"""One recipe per figure kind, in the order group figures are planned and drawn."""


def group_recipe(kind: FigureKind) -> GroupFigureRecipe:
    """The recipe of ``kind``, which must be drawn from a group report."""
    match FIGURE_RECIPES[kind]:
        case GroupFigureRecipe() as recipe:
            return recipe
        case HistoryFigureRecipe():
            raise NotAGroupKindError(f"{kind.value} is drawn from a stored method")
        case unreachable:
            assert_never(unreachable)


def history_variant(kind: FigureKind) -> HistoryVariant:
    """The one figure of ``kind``, which must be drawn from stored method histories."""
    match FIGURE_RECIPES[kind]:
        case HistoryFigureRecipe(variant=variant):
            return variant
        case GroupFigureRecipe():
            raise NotAHistoryKindError(f"{kind.value} is drawn from a group report")
        case unreachable:
            assert_never(unreachable)


def table_of(figure: str, kind: FigureKind) -> FigureTable:
    """The table beside ``figure`` of ``kind``: its recipe's kind of table, if it writes one."""
    match FIGURE_RECIPES[kind]:
        case GroupFigureRecipe(table=table_kind):
            return SiblingTable(f"{figure.removesuffix('.png')}{TABLE_SUFFIX}", table_kind)
        case HistoryFigureRecipe():
            return NoTable()
        case unreachable:
            assert_never(unreachable)


def plan_figure[M: FigureMetadata](
    variant: FigureVariant | HistoryVariant, metadata: M, *, slug: str
) -> FigurePlan[M]:
    """The plan of ``variant`` for the report subject named by ``slug``."""
    return FigurePlan(variant.filename(slug), variant.title, metadata)


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
        for recipe in _group_recipes()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )


def figure_summaries(outcome: GroupReport) -> Mapping[str, pl.DataFrame]:
    """The frame each figure planned for ``outcome`` draws and stores, by variant stem.

    Each frame is computed once, here, and is both what the drawer draws and the table written
    beside the figure. A figure that compares against the reference has no frame, and so is not
    drawn, when the group has no configured reference among its methods.
    """
    reference = reference_among_methods(outcome.observations, outcome.reference_selection)
    frames = (
        (variant.stem, _summary(variant, outcome, reference))
        for recipe in _group_recipes()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )
    return {stem: frame for stem, frame in frames if frame is not None}


def _summary(
    variant: FigureVariant, outcome: GroupReport, reference: PreconditionerName | None
) -> pl.DataFrame | None:
    observations = outcome.observations
    match variant.statistic:
        case Statistic.CONVERGED_COUNT:
            return summarize(observations, variant.metric, variant.population)
        case Statistic.WORKLOAD_TOTAL:
            return build_group_table(outcome.summaries)
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
        case unreachable:
            assert_never(unreachable)


def _group_recipes() -> tuple[GroupFigureRecipe, ...]:
    return tuple(
        recipe for recipe in FIGURE_RECIPES.values() if isinstance(recipe, GroupFigureRecipe)
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
        method_source(
            stored[name].run_id, artifacts.setup_events, artifacts.solve_attempts, artifacts.summary
        )
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
        case (
            Statistic.CONVERGED_COUNT
            | Statistic.WORKLOAD_TOTAL
            | Statistic.ECDF
            | Statistic.PERFORMANCE_PROFILE
            | Statistic.PAIRED_VALUES
            | Statistic.CUMULATIVE_COST
        ):
            return NoInterval()
        case unreachable:
            assert_never(unreachable)
