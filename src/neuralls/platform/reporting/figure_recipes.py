"""The figure registry: one recipe per kind, its variants and the lookups over them."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import assert_never

from neuralls.domain.solver.models.group_outcomes import GroupReport
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import (
    Population,
)
from neuralls.platform.reporting import group_drawers
from neuralls.platform.reporting.cohort_variants import cohort_level, cohort_variants
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    Statistic,
    TableKind,
)
from neuralls.platform.reporting.group_drawers import GroupDrawer
from neuralls.platform.reporting.seaborn_means import BarStatistic, bar_statistic_text
from neuralls.shared.types import PreconditionerName


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


@dataclass(frozen=True, slots=True)
class GroupFigureRecipe:
    """Everything fixed about a group figure kind.

    ``draw`` is the drawer of every variant, ``table`` the kind of table stored beside each
    figure, which also decides the report levels the kind is drawn at (``cohort_variants.py``).
    """

    kind: FigureKind
    variants: tuple[FigureVariant, ...]
    failure_treatment: FailureTreatment
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


@dataclass(frozen=True, slots=True)
class _Time:
    """One time drawn: its file-stem word, its titles and its metric.

    ``word`` is a stage of the cost vocabulary in ``domain/solver/README.md``, with the
    ``_amortized`` modifier for the multi-RHS variant; ``short`` names it inside a longer title.
    """

    word: str
    title: str
    short: str
    metric: MetricId


_SOLVE = _Time("solve", "Solve-only Time per Pair", "Solve-only", MetricId.SOLVE_S)
_ONLINE = _Time("online", "Online Time per Pair", "Online", MetricId.ONLINE_S)
_ONLINE_AMORTIZED = _Time(
    "online_amortized",
    "Amortized Online Time per Pair",
    "Amortized Online",
    MetricId.ONLINE_AMORTIZED_S,
)
_TIMES = (_SOLVE, _ONLINE, _ONLINE_AMORTIZED)
_ONLINE_TIMES = (_ONLINE, _ONLINE_AMORTIZED)
"""The pair online time and its amortized multi-RHS variant, drawn side by side by each kind
that compares online costs."""


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
            FigureKind.SOLVE_TIME,
            (
                FigureVariant(
                    "workload_solve_bars",
                    "Solve-only Workload Sum",
                    "sum over planned pairs of solve-only seconds, including failures; a pair "
                    "with no solve adds none",
                    MetricId.SOLVE_S,
                    Population.OWN,
                    Statistic.WORKLOAD_SUM,
                ),
            ),
            FailureTreatment.NONE,
            group_drawers.workload_bars,
            TableKind.WORKLOAD,
        ),
        GroupFigureRecipe(
            FigureKind.ONLINE_TIME,
            (
                FigureVariant(
                    "workload_online_bars",
                    "Online Workload Sum",
                    "sum over planned pairs of the matrix setup seconds charged to each pair "
                    "plus its solve-only seconds; a failed setup is still charged to each "
                    "planned pair of its matrix, so this exceeds the sum of pair online times",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.WORKLOAD_SUM,
                ),
                FigureVariant(
                    "workload_online_amortized_bars",
                    "Amortized Online Workload Sum",
                    "sum over planned pairs of each pair's share of its matrix setup seconds "
                    "(setup over the matrix's planned RHS) plus its solve-only seconds",
                    MetricId.ONLINE_AMORTIZED_S,
                    Population.OWN,
                    Statistic.WORKLOAD_SUM,
                ),
            ),
            FailureTreatment.NONE,
            group_drawers.workload_bars,
            TableKind.WORKLOAD,
        ),
        GroupFigureRecipe(
            FigureKind.PAIRED_TIME,
            tuple(
                FigureVariant(
                    f"paired_{time.word}_scatter",
                    time.title,
                    f"{METRICS[time.metric].label} of each pair against the reference; "
                    "asymmetric failures axis-censored; both-failed pairs omitted",
                    time.metric,
                    Population.OWN,
                    Statistic.PAIRED_VALUES,
                )
                for time in _ONLINE_TIMES
            ),
            FailureTreatment.CENSORED,
            group_drawers.paired_time_scatter,
            TableKind.PAIRED,
        ),
        GroupFigureRecipe(
            FigureKind.SPEED_RATIO,
            tuple(
                FigureVariant(
                    f"speed_ratio_{time.word}_cdf",
                    f"{time.short} Speed Ratio",
                    f"reference/method ratio of {METRICS[time.metric].label}",
                    time.metric,
                    Population.OWN,
                    Statistic.SPEED_RATIO_CDF,
                )
                for time in _ONLINE_TIMES
            ),
            FailureTreatment.OMITTED,
            group_drawers.speed_ratio_cdf,
            TableKind.SPEED_RATIO,
        ),
        GroupFigureRecipe(
            FigureKind.PERFORMANCE_PROFILE,
            tuple(
                FigureVariant(
                    f"performance_profile_{time.word}",
                    f"{time.short} Performance Profile",
                    f"{METRICS[time.metric].label} performance ratio",
                    time.metric,
                    Population.OWN,
                    Statistic.PERFORMANCE_PROFILE,
                )
                for time in _TIMES
            ),
            FailureTreatment.CENSORED,
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
            group_drawers.ecdf_iterations,
            TableKind.SERIES,
        ),
        GroupFigureRecipe(
            FigureKind.ECDF_TIME,
            tuple(
                FigureVariant(
                    f"ecdf_{time.word}",
                    time.title,
                    f"{METRICS[time.metric].label} of each planned pair",
                    time.metric,
                    Population.OWN,
                    Statistic.ECDF,
                )
                for time in _TIMES
            ),
            FailureTreatment.CENSORED,
            group_drawers.ecdf_time,
            TableKind.SERIES,
        ),
        GroupFigureRecipe(
            FigureKind.BREAKEVEN,
            (
                FigureVariant(
                    "breakeven_online_amortized",
                    "Break-even",
                    "mean setup + n x mean solve-only seconds per RHS on one matrix, means over the "
                    "matrices and systems common to the method and the reference",
                    MetricId.ONLINE_AMORTIZED_S,
                    Population.COMMON,
                    Statistic.CUMULATIVE_COST,
                ),
            ),
            FailureTreatment.OMITTED,
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
            group_drawers.mean_over_systems_bars,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.TIME_MEANS,
            tuple(
                _bar_variant(
                    f"time_means_{time.word}_{population}",
                    time.title,
                    time.metric,
                    population,
                    Statistic.MEAN_OVER_SYSTEMS,
                )
                for time in _TIMES
                for population in Population
            ),
            FailureTreatment.OMITTED,
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
            group_drawers.time_per_iteration,
            TableKind.COHORT_RATIO_OF_SUMS,
        ),
        GroupFigureRecipe(
            FigureKind.ITERATION_DISTRIBUTION,
            (_distribution_variant("iteration_distribution", "Iteration", MetricId.ITERATIONS),),
            FailureTreatment.OMITTED,
            group_drawers.distribution,
            TableKind.COHORT_SUMMARY,
        ),
        GroupFigureRecipe(
            FigureKind.TIME_DISTRIBUTION,
            tuple(
                _distribution_variant(f"time_distribution_{time.word}", time.title, time.metric)
                for time in _TIMES
            ),
            FailureTreatment.OMITTED,
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
            group_drawers.matrix_heatmap,
            TableKind.MATRIX_BREAKDOWN,
        ),
        GroupFigureRecipe(
            FigureKind.FULL_COST_BREAKDOWN,
            (
                FigureVariant(
                    "full_cost_breakdown",
                    "Full Cost",
                    "offline (dataset generation + fit, paid once per method artifact) "
                    "+ setup + solve-only seconds of the report's pairs, each pair charged its "
                    "matrix's setup, stacked per method; a fit is charged the share of the "
                    "generation in the folds it read (train for a POD fit, train and validation "
                    "for neural training); an offline stage with no recorded cost draws no "
                    "segment and leaves the full cost unstated",
                    MetricId.ONLINE_S,
                    Population.OWN,
                    Statistic.FULL_COST,
                ),
            ),
            FailureTreatment.NONE,
            group_drawers.full_cost_breakdown,
            TableKind.FULL_COST_BREAKDOWN,
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


def compares_to_reference(statistic: Statistic) -> bool:
    """Whether a figure drawing ``statistic`` compares every method against the reference."""
    match statistic:
        case Statistic.PAIRED_VALUES | Statistic.SPEED_RATIO_CDF | Statistic.CUMULATIVE_COST:
            return True
        case (
            Statistic.WORKLOAD_SUM
            | Statistic.ECDF
            | Statistic.PERFORMANCE_PROFILE
            | Statistic.MEAN_AND_MEDIAN
            | Statistic.MEAN_OVER_SYSTEMS
            | Statistic.MEAN_OVER_MATRICES
            | Statistic.RATIO_OF_SUMS
            | Statistic.QUANTILES
            | Statistic.MEAN_PER_MATRIX
            | Statistic.FULL_COST
        ):
            return False
        case unreachable:
            assert_never(unreachable)


def drawn_variants(
    report: GroupReport, reference: PreconditionerName | None
) -> Iterator[tuple[GroupFigureRecipe, FigureVariant]]:
    """Every group figure ``report`` gets, in registry order, with the recipe it belongs to.

    A kind its report level does not draw (``cohort_variants``: a raw-scale aggregate at a mixture
    level) is skipped. ``reference`` is the group's reference among its methods
    (``reference_among_methods``); with none, a figure that compares against it is not drawn, so
    it is neither planned nor summarized.
    """
    level = cohort_level(report.label)
    for recipe in group_recipes():
        if not cohort_variants(recipe.table, level):
            continue
        for variant in recipe.variants:
            if reference is None and compares_to_reference(variant.statistic):
                continue
            yield recipe, variant


def drawn_reference(
    variant: FigureVariant, reference: PreconditionerName | None
) -> PreconditionerName:
    """The reference a figure of ``drawn_variants`` compares against, which it has by construction.

    Raises:
        ValueError: ``variant`` was not given by ``drawn_variants`` for this ``reference``.
    """
    match reference:
        case None:
            raise ValueError(f"{variant.stem} compares against a reference the group lacks")
        case name:
            return name
