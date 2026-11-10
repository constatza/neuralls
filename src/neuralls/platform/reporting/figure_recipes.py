"""The figure registry: one recipe per kind, its variants and the lookups over them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import (
    Population,
)
from neuralls.platform.reporting import group_drawers
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    Statistic,
    TableKind,
)
from neuralls.platform.reporting.group_drawers import GroupDrawer
from neuralls.platform.reporting.seaborn_means import BarStatistic, bar_statistic_text


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


def group_recipes() -> tuple[GroupFigureRecipe, ...]:
    return tuple(
        recipe for recipe in FIGURE_RECIPES.values() if isinstance(recipe, GroupFigureRecipe)
    )
