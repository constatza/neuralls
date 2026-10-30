"""Pre-draw planning of group figures: file names, kinds and scientific metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

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
    TimingScenario,
    make_plan,
    method_source,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName


@dataclass(frozen=True, slots=True)
class _FigureSpec:
    filename: str
    kind: FigureKind
    scenario: TimingScenario | None = None


def figure_plans(
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    presentation: GroupPresentation,
    file_key: str,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> tuple[FigurePlan, ...]:
    """Build every figure's scientific context before drawing.

    ``stored`` holds the method run of each method ``presentation`` lists.
    """
    specs = _planned_figure_specs(outcome, file_key)
    plans: list[FigurePlan] = []
    for spec in specs:
        plan = make_plan(
            spec.kind,
            _figure_metadata(spec.kind, spec.scenario, outcome, presentation, stored),
            group_slug=file_key,
        )
        plans.append(replace(plan, filename=spec.filename))
    return tuple(plans)


def _planned_figure_specs(
    outcome: RecipeProfileReport | WorkloadMixtureReport, file_key: str
) -> tuple[_FigureSpec, ...]:
    aggregate = _FigureSpec(f"aggregate_bars_{file_key}.png", FigureKind.AGGREGATE_BARS)
    solve_bar = _FigureSpec(
        f"solve_time_bars_{file_key}.png", FigureKind.SOLVE_TIME, TimingScenario.SOLVE
    )
    online_bar = _FigureSpec(
        f"online_time_bars_{file_key}.png", FigureKind.ONLINE_TIME, TimingScenario.ONLINE
    )
    if isinstance(outcome, RecipeProfileReport):
        return (
            aggregate,
            solve_bar,
            online_bar,
            _FigureSpec(
                f"paired_time_scatter_{file_key}.png",
                FigureKind.PAIRED_TIME,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(
                f"speed_ratio_cdf_{file_key}.png",
                FigureKind.SPEED_RATIO,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(
                f"performance_profile_solve_{file_key}.png",
                FigureKind.PERFORMANCE_PROFILE,
                TimingScenario.SOLVE,
            ),
            _FigureSpec(
                f"performance_profile_online_{file_key}.png",
                FigureKind.PERFORMANCE_PROFILE,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(f"ecdf_iterations_{file_key}.png", FigureKind.ECDF_ITERATIONS),
            _FigureSpec(
                f"ecdf_time_solve_{file_key}.png",
                FigureKind.ECDF_TIME,
                TimingScenario.SOLVE,
            ),
            _FigureSpec(
                f"ecdf_time_online_{file_key}.png",
                FigureKind.ECDF_TIME,
                TimingScenario.ONLINE,
            ),
        )
    return (aggregate, solve_bar, online_bar)


def _figure_metadata(
    kind: FigureKind,
    scenario: TimingScenario | None,
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    presentation: GroupPresentation,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> FigureMetadata:
    dataset_ids = tuple(str(dataset) for dataset in outcome.dataset_membership)
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
    metric = {
        FigureKind.AGGREGATE_BARS: "converged systems",
        FigureKind.SOLVE_TIME: "solve seconds (workload total)",
        FigureKind.ONLINE_TIME: "workload online seconds (setup + solve total)",
        FigureKind.PAIRED_TIME: (
            "amortized online seconds per paired RHS; asymmetric failures axis-censored; "
            "both-failed pairs omitted"
        ),
        FigureKind.PERFORMANCE_PROFILE: (
            "solve-seconds performance ratio"
            if scenario is TimingScenario.SOLVE
            else "amortized-online-seconds performance ratio"
        ),
        FigureKind.ECDF_ITERATIONS: "iterations",
        FigureKind.ECDF_TIME: (
            "solve seconds per planned RHS"
            if scenario is TimingScenario.SOLVE
            else "amortized online seconds per planned RHS"
        ),
        FigureKind.SPEED_RATIO: "reference/method amortized-online speed ratio",
    }.get(kind, kind.value)
    failure_treatment = (
        FailureTreatment.NONE
        if kind in {FigureKind.SOLVE_TIME, FigureKind.ONLINE_TIME}
        else FailureTreatment.OMITTED
        if kind is FigureKind.SPEED_RATIO
        else FailureTreatment.CENSORED
    )
    aggregation_rule = (
        "sum observed setup and solve event costs; recipe contributions follow planned RHS membership"
        if isinstance(outcome, WorkloadMixtureReport)
        else "one observation per planned (matrix, RHS) system"
    )
    return FigureMetadata(
        kind=kind,
        metric_definition=metric,
        scenario=scenario,
        report_level=level_name(outcome.label),
        scope=presentation.scope,
        dataset_ids=dataset_ids,
        recipes=recipes,
        n_matrices=outcome.n_matrices,
        n_planned_rhs=outcome.n_planned_rhs,
        reference_selection=reference,
        failure_treatment=failure_treatment,
        timing_policy="setup once per dataset/matrix + solve per planned RHS",
        recipe_counts=outcome.recipe_counts,
        aggregation_rule=aggregation_rule,
        recipe_weights=weights,
        solver_contexts=presentation.solver_contexts,
        methods=methods,
        sources=sources,
    )
