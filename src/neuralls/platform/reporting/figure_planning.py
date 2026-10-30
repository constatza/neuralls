"""Pre-draw planning of group figures: file names, kinds and scientific metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.report_levels import (
    level_name,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigurePlan,
    GroupAggregationRule,
    GroupFigureMetadata,
    TimingPolicy,
    TimingScenario,
    method_source,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName

type GroupReport = RecipeProfileReport | WorkloadMixtureReport


@dataclass(frozen=True, slots=True)
class FigureVariant:
    """One figure a kind produces: its timing scenario, file stem, title and measured quantity."""

    scenario: TimingScenario | None
    stem: str
    title: str
    metric_definition: str

    def filename(self, slug: str) -> str:
        """The file name of this figure for the report subject named by ``slug``."""
        return f"{self.stem}_{slug}.png"


@dataclass(frozen=True, slots=True)
class FigureRecipe:
    """Everything fixed about a figure kind: its variants, failure handling and where it applies.

    ``applies_to`` lists the group report types the kind is planned for; method kinds list
    none because they are drawn from a stored method, not from a group outcome.
    """

    kind: FigureKind
    variants: tuple[FigureVariant, ...]
    failure_treatment: FailureTreatment
    applies_to: tuple[type[GroupReport], ...]

    def variant(self, scenario: TimingScenario | None) -> FigureVariant:
        """The variant of this kind for ``scenario``."""
        for variant in self.variants:
            if variant.scenario is scenario:
                return variant
        raise UnknownFigureVariantError(f"{self.kind.value} has no variant for scenario {scenario}")


class UnknownFigureVariantError(LookupError):
    """A kind was asked for a scenario it has no figure for."""


_BOTH: tuple[type[GroupReport], ...] = (RecipeProfileReport, WorkloadMixtureReport)
_PROFILE: tuple[type[GroupReport], ...] = (RecipeProfileReport,)

FIGURE_RECIPES: Mapping[FigureKind, FigureRecipe] = {
    recipe.kind: recipe
    for recipe in (
        FigureRecipe(
            FigureKind.AGGREGATE_BARS,
            (FigureVariant(None, "aggregate_bars", "Convergence", "converged systems"),),
            FailureTreatment.CENSORED,
            _BOTH,
        ),
        FigureRecipe(
            FigureKind.SOLVE_TIME,
            (
                FigureVariant(
                    TimingScenario.SOLVE,
                    "solve_time_bars",
                    "Solve Time",
                    "solve seconds (workload total)",
                ),
            ),
            FailureTreatment.NONE,
            _BOTH,
        ),
        FigureRecipe(
            FigureKind.ONLINE_TIME,
            (
                FigureVariant(
                    TimingScenario.ONLINE,
                    "online_time_bars",
                    "Online Time",
                    "workload online seconds (setup + solve total)",
                ),
            ),
            FailureTreatment.NONE,
            _BOTH,
        ),
        FigureRecipe(
            FigureKind.PAIRED_TIME,
            (
                FigureVariant(
                    TimingScenario.ONLINE,
                    "paired_time_scatter",
                    "Online Time",
                    "amortized online seconds per paired RHS; asymmetric failures "
                    "axis-censored; both-failed pairs omitted",
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
        ),
        FigureRecipe(
            FigureKind.SPEED_RATIO,
            (
                FigureVariant(
                    TimingScenario.ONLINE,
                    "speed_ratio_cdf",
                    "Speed Ratio",
                    "reference/method amortized-online speed ratio",
                ),
            ),
            FailureTreatment.OMITTED,
            _PROFILE,
        ),
        FigureRecipe(
            FigureKind.PERFORMANCE_PROFILE,
            (
                FigureVariant(
                    TimingScenario.SOLVE,
                    "performance_profile_solve",
                    "Performance Profile",
                    "solve-seconds performance ratio",
                ),
                FigureVariant(
                    TimingScenario.ONLINE,
                    "performance_profile_online",
                    "Performance Profile",
                    "amortized-online-seconds performance ratio",
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
        ),
        FigureRecipe(
            FigureKind.ECDF_ITERATIONS,
            (FigureVariant(None, "ecdf_iterations", "Iterations", "iterations"),),
            FailureTreatment.CENSORED,
            _PROFILE,
        ),
        FigureRecipe(
            FigureKind.ECDF_TIME,
            (
                FigureVariant(
                    TimingScenario.SOLVE,
                    "ecdf_time_solve",
                    "Solve Time",
                    "solve seconds per planned RHS",
                ),
                FigureVariant(
                    TimingScenario.ONLINE,
                    "ecdf_time_online",
                    "Online Time",
                    "amortized online seconds per planned RHS",
                ),
            ),
            FailureTreatment.CENSORED,
            _PROFILE,
        ),
        FigureRecipe(
            FigureKind.RESIDUAL_HISTORY,
            (
                FigureVariant(
                    None,
                    "residual_history",
                    "Residual History",
                    "relative residual by iteration",
                ),
            ),
            FailureTreatment.NONE,
            (),
        ),
        FigureRecipe(
            FigureKind.ENERGY_HISTORY,
            (
                FigureVariant(
                    None,
                    "energy_history",
                    "Energy History",
                    "relative energy error by iteration",
                ),
            ),
            FailureTreatment.NONE,
            (),
        ),
    )
}
"""One recipe per figure kind, in the order group figures are planned and drawn."""


def plan_figure[M: FigureMetadata](
    variant: FigureVariant, metadata: M, *, slug: str
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
        for recipe in FIGURE_RECIPES.values()
        if isinstance(outcome, recipe.applies_to)
        for variant in recipe.variants
    )


def _figure_metadata(
    recipe: FigureRecipe,
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
        scenario=variant.scenario,
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
    )
