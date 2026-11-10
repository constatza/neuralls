"""The figure registry: one recipe per kind, and the pre-draw plan of every group figure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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
from neuralls.domain.solver.models.cost import OfflineStages
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.cohorts import CohortRows, by_cohort
from neuralls.domain.solver.reporting.matrix_breakdown import matrix_breakdown
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.report_levels import (
    level_name,
)
from neuralls.domain.solver.reporting.summary import (
    RATIO_OF_SUMS_SCHEMA,
    SUMMARY_SCHEMA,
    Population,
    summarize,
    time_per_iteration_summary,
)
from neuralls.platform.reporting import group_drawers
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
from neuralls.platform.reporting.figure_table import build_cost_breakdown_table, build_group_table
from neuralls.platform.reporting.group_drawers import GroupDrawer
from neuralls.platform.reporting.seaborn_means import BarStatistic, bar_statistic_text
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName

TABLE_SUFFIX = ".parquet"
"""A figure's table shares its stem and directory; only the suffix differs."""


def variant_stem(stem: str, slug: str) -> str:
    """The file stem of the figure of kind-level ``stem`` for the report subject ``slug``."""
    return f"{stem}_{slug}"


@dataclass(frozen=True, slots=True)
class FigureVariant:
    """One group figure a kind produces, and the statistic of a metric it draws.

    ``population`` says which systems the metric is taken over and ``statistic`` what is drawn
    of it; a kind's variants differ in ``metric`` or ``population``, the pair that tells them
    apart.
    """

    stem: str
    title: str
    metric_definition: str
    metric: MetricId
    population: Population
    statistic: Statistic

    def figure_stem(self, slug: str) -> str:
        """The file stem of this figure for the report subject named by ``slug``."""
        return variant_stem(self.stem, slug)


@dataclass(frozen=True, slots=True)
class HistoryVariant:
    """The one figure a convergence-history kind produces; it summarizes no metric."""

    stem: str
    title: str
    metric_definition: str

    def figure_stem(self, slug: str) -> str:
        """The file stem of this figure for the report subject named by ``slug``."""
        return variant_stem(self.stem, slug)


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
        """Require the variants to differ in metric or population, which select the one to draw."""
        keys = [(variant.metric, variant.population) for variant in self.variants]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.kind.value} has two variants of one metric and population")

    def variant(self, metric: MetricId, population: Population) -> FigureVariant:
        """The variant of this kind that summarizes ``metric`` over ``population``."""
        for variant in self.variants:
            if (variant.metric, variant.population) == (metric, population):
                return variant
        raise UnknownFigureVariantError(
            f"{self.kind.value} has no variant for metric {metric} over population {population}"
        )


@dataclass(frozen=True, slots=True)
class HistoryFigureRecipe:
    """Everything fixed about a convergence-history kind, drawn from a stored method."""

    kind: FigureKind
    variant: HistoryVariant
    failure_treatment: FailureTreatment


type FigureRecipe = GroupFigureRecipe | HistoryFigureRecipe

_TIME_MEAN_METRICS: tuple[tuple[str, str, MetricId], ...] = (
    ("solve", "Solve Time", MetricId.SOLVE_S),
    ("online", "Online Time", MetricId.ONLINE_S),
    ("standalone", "Standalone Time", MetricId.STANDALONE_S),
)
"""File-stem word, title and metric of each time whose mean is drawn."""


def _bar_variant(
    stem: str, title: str, metric: MetricId, population: Population, statistic: BarStatistic
) -> FigureVariant:
    """A bar variant whose definition is built from the metric catalog's label."""
    return FigureVariant(
        stem,
        f"{title}{_population_title(population)}",
        f"{bar_statistic_text(statistic)} of {METRICS[metric].label}, {population} population",
        metric,
        population,
        statistic,
    )


_DISTRIBUTION_SUMMARY = (
    "box at q25, q50, q75 and whiskers from q05 to q95 of the systems, a distribution summary "
    "and not a confidence interval; points are per-matrix means"
)
"""What a distribution's marks are; its card records no interval, so this says why."""

_DISTRIBUTION_TIMES: tuple[tuple[str, str, MetricId], ...] = (
    ("solve", "Solve Time", MetricId.SOLVE_S),
    ("online", "Online Time", MetricId.ONLINE_S),
)
"""File-stem word, title and metric of each time whose distribution is drawn."""


def _distribution_variant(stem: str, title: str, metric: MetricId) -> FigureVariant:
    """A distribution over each method's own systems, whose quantiles the summary table holds."""
    population = Population.OWN
    return FigureVariant(
        stem,
        f"{title} Distribution",
        f"distribution of {METRICS[metric].label} over systems, {population} population; "
        f"{_DISTRIBUTION_SUMMARY}",
        metric,
        population,
        Statistic.QUANTILES,
    )


def _population_title(population: Population) -> str:
    match population:
        case Population.OWN:
            return ""
        case Population.COMMON:
            return " (Common Systems)"
        case unreachable:
            assert_never(unreachable)


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
            group_drawers.aggregate_bars,
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
            group_drawers.solve_time_bars,
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
            group_drawers.online_time_bars,
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
            group_drawers.paired_time_scatter,
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
            group_drawers.speed_ratio_cdf,
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
            group_drawers.performance_profile,
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
            group_drawers.ecdf_iterations,
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
            group_drawers.ecdf_time,
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
            group_drawers.break_even,
            TableKind.COST_CURVES,
        ),
        GroupFigureRecipe(
            FigureKind.CONVERGENCE_FACTOR,
            (
                FigureVariant(
                    "convergence_factor",
                    "Convergence Factor",
                    "residual reduction factor per iteration (r_J / r_0)^(1/J) of each converged "
                    "system's stored history",
                    MetricId.CONVERGENCE_FACTOR,
                    Population.OWN,
                    Statistic.MEAN_AND_MEDIAN,
                ),
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.convergence_factors,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.ITERATION_MEANS,
            (
                _bar_variant(
                    "iteration_means",
                    "Iterations",
                    MetricId.ITERATIONS,
                    Population.OWN,
                    Statistic.MEAN_OVER_SYSTEMS,
                ),
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.mean_over_systems_bars,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.TIME_MEANS,
            tuple(
                _bar_variant(
                    f"time_means_{word}_{population}",
                    title,
                    metric,
                    population,
                    Statistic.MEAN_OVER_SYSTEMS,
                )
                for word, title, metric in _TIME_MEAN_METRICS
                for population in Population
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.mean_over_systems_bars,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.SETUP_MEANS,
            (
                _bar_variant(
                    "setup_means",
                    "Setup Time",
                    MetricId.SETUP_S,
                    Population.OWN,
                    Statistic.MEAN_OVER_MATRICES,
                ),
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.mean_over_matrices_bars,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.TIME_PER_ITERATION,
            (
                _bar_variant(
                    "time_per_iteration",
                    "Time per Iteration",
                    MetricId.TIME_PER_ITERATION_S,
                    Population.OWN,
                    Statistic.RATIO_OF_SUMS,
                ),
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.time_per_iteration,
            TableKind.COHORT_RATIO_OF_SUMS,
        ),
        GroupFigureRecipe(
            FigureKind.ITERATION_DISTRIBUTION,
            (_distribution_variant("iteration_distribution", "Iteration", MetricId.ITERATIONS),),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.distribution,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.TIME_DISTRIBUTION,
            tuple(
                _distribution_variant(f"time_distribution_{word}", title, metric)
                for word, title, metric in _DISTRIBUTION_TIMES
            ),
            FailureTreatment.OMITTED,
            _BOTH,
            group_drawers.distribution,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.MATRIX_HEATMAP,
            (
                FigureVariant(
                    "matrix_heatmap_iterations",
                    "Iterations per Matrix",
                    f"mean of {METRICS[MetricId.ITERATIONS].label} over the converged RHS of "
                    "each matrix, per method; a hatched cell has no converged RHS and is "
                    "labelled 0/r solved, r the planned RHS of the matrix",
                    MetricId.ITERATIONS,
                    Population.OWN,
                    Statistic.MEAN_PER_MATRIX,
                ),
            ),
            FailureTreatment.OMITTED,
            # A mixture level would pool recipes inside a matrix's cell; the recipe levels below
            # it already draw every matrix once per recipe.
            _PROFILE,
            group_drawers.matrix_heatmap,
            TableKind.MATRIX_BREAKDOWN,
        ),
        GroupFigureRecipe(
            FigureKind.COST_BREAKDOWN,
            (
                FigureVariant(
                    "cost_breakdown",
                    "Total Cost",
                    "offline (dataset generation + training, paid once per method artifact) "
                    "+ setup + solve seconds of the report's workload, stacked per method; an "
                    "offline stage with no recorded cost draws no segment and leaves the total "
                    "unstated",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.OFFLINE_ONLINE_TOTAL,
                ),
            ),
            FailureTreatment.NONE,
            _BOTH,
            group_drawers.cost_breakdown,
            TableKind.COST_BREAKDOWN,
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
        for recipe in _group_recipes()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )


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
        for recipe in _group_recipes()
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
