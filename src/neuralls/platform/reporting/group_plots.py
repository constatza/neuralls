"""Drawing of the group outcome plots from their pre-draw figure plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from neuralls.domain.benchmarking.costs import amortized_online_seconds, solve_seconds
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
)
from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.domain.solver.models.statistics import MethodComparison, PairedComparison
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
    FigurePlan,
    TimingScenario,
)
from neuralls.platform.reporting.figure_text import (
    method_legend,
)
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_aggregate_bars,
    plot_ecdf_profile,
    plot_online_time_bars,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speed_ratio_cdf,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName


@dataclass(frozen=True, slots=True)
class GroupStyler:
    """Turns the pre-draw plan of a figure into the style its plot draws with.

    The plan is the only source of a figure's fixed title and scope; the legend names
    every method of the group from its structured presentation.
    """

    presentation: GroupPresentation
    notation: NotationPolicy

    def style(self, plan: FigurePlan) -> GroupStyle:
        """The style for ``plan``: fixed title, rendered scope, legend and series styling."""
        maps = plot_style_maps(self.presentation.plot_styles)
        return GroupStyle(
            title=plan.title,
            scope=plan.metadata.scope,
            notation=self.notation,
            legend=method_legend(
                {name: method.compact for name, method in self.presentation.methods.items()},
                self.notation,
            ),
            families=maps.families,
            color_keys=maps.color_keys,
            marker_keys=maps.marker_keys,
        )


def _plan_for(
    plans: Sequence[FigurePlan], kind: FigureKind, scenario: TimingScenario | None = None
) -> FigurePlan:
    return next(
        plan for plan in plans if plan.metadata.kind is kind and plan.metadata.scenario is scenario
    )


def _paired_by_method(
    comparisons: Mapping[PreconditionerName, MethodComparison],
) -> dict[PreconditionerName, PairedComparison]:
    """The paired comparisons of the methods that had any system solved alongside the reference."""
    return {name: comparison.paired for name, comparison in comparisons.items()}


def _write_profile_plots(
    outcome: RecipeProfileReport,
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> list[Path | None]:
    """Draw fine-only iteration and separate solve/online time profiles."""
    performance_solve = _plan_for(plans, FigureKind.PERFORMANCE_PROFILE, TimingScenario.SOLVE)
    performance_online = _plan_for(plans, FigureKind.PERFORMANCE_PROFILE, TimingScenario.ONLINE)
    ecdf_iterations = _plan_for(plans, FigureKind.ECDF_ITERATIONS)
    ecdf_solve = _plan_for(plans, FigureKind.ECDF_TIME, TimingScenario.SOLVE)
    ecdf_online = _plan_for(plans, FigureKind.ECDF_TIME, TimingScenario.ONLINE)
    return [
        plot_performance_profile(
            outcome.outcomes,
            style=styler.style(performance_solve),
            cost=solve_seconds,
            save_path=directory / performance_solve.filename,
        ),
        plot_performance_profile(
            outcome.outcomes,
            style=styler.style(performance_online),
            cost=amortized_online_seconds,
            save_path=directory / performance_online.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_iterations),
            save_path=directory / ecdf_iterations.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_solve),
            cost=solve_seconds,
            xlabel="Solve time per system [s]",
            save_path=directory / ecdf_solve.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_online),
            cost=amortized_online_seconds,
            xlabel="Amortized online time per system [s]",
            save_path=directory / ecdf_online.filename,
        ),
    ]


def _write_system_plots(
    outcome: RecipeProfileReport,
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> list[Path | None]:
    """Draw setup-aware timing bars, paired timing, and ratio plots."""
    paired_plan = _plan_for(plans, FigureKind.PAIRED_TIME, TimingScenario.ONLINE)
    solve_plan = _plan_for(plans, FigureKind.SOLVE_TIME, TimingScenario.SOLVE)
    online_plan = _plan_for(plans, FigureKind.ONLINE_TIME, TimingScenario.ONLINE)
    ratio_plan = _plan_for(plans, FigureKind.SPEED_RATIO, TimingScenario.ONLINE)
    paired = (
        plot_paired_time_scatter(
            outcome.outcomes,
            reference=outcome.reference_selection,
            style=styler.style(paired_plan),
            save_path=directory / paired_plan.filename,
        )
        if isinstance(outcome.reference_selection, ConfiguredReference)
        else None
    )
    plots: list[Path | None] = [
        plot_online_time_bars(
            outcome.summaries,
            scenario="solve",
            style=styler.style(solve_plan),
            save_path=directory / solve_plan.filename,
        ),
        plot_online_time_bars(
            outcome.summaries,
            scenario="online",
            style=styler.style(online_plan),
            save_path=directory / online_plan.filename,
        ),
        paired,
        plot_speed_ratio_cdf(
            _paired_by_method(outcome.comparisons),
            style=styler.style(ratio_plan),
            reference_name=outcome.reference_selection.name
            if isinstance(outcome.reference_selection, ConfiguredReference)
            else None,
            save_path=directory / ratio_plan.filename,
        ),
    ]
    return plots


def write_outcome_plots(
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> tuple[Path, ...]:
    """Draw plots allowed by the typed report variant."""
    match outcome:
        case RecipeProfileReport():
            aggregate_plan = _plan_for(plans, FigureKind.AGGREGATE_BARS)
            aggregate = plot_aggregate_bars(
                outcome,
                style=styler.style(aggregate_plan),
                save_path=directory / aggregate_plan.filename,
            )
            drawn = [
                aggregate,
                *_write_system_plots(outcome, styler, directory, plans),
                *_write_profile_plots(outcome, styler, directory, plans),
            ]
        case WorkloadMixtureReport():
            aggregate_plan = _plan_for(plans, FigureKind.AGGREGATE_BARS)
            solve_plan = _plan_for(plans, FigureKind.SOLVE_TIME, TimingScenario.SOLVE)
            online_plan = _plan_for(plans, FigureKind.ONLINE_TIME, TimingScenario.ONLINE)
            aggregate = plot_aggregate_bars(
                outcome,
                style=styler.style(aggregate_plan),
                save_path=directory / aggregate_plan.filename,
            )
            drawn = [
                aggregate,
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="solve",
                    style=styler.style(solve_plan),
                    save_path=directory / solve_plan.filename,
                ),
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="online",
                    style=styler.style(online_plan),
                    save_path=directory / online_plan.filename,
                ),
            ]
        case _ as unreachable:
            assert_never(unreachable)
    return tuple(path for path in drawn if path is not None)
