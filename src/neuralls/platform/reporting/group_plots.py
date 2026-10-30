"""Drawing of the group outcome plots from their pre-draw figure plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from neuralls.domain.benchmarking.costs import Cost, amortized_online_seconds, solve_seconds
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
    GroupFigureMetadata,
    TimingScenario,
)
from neuralls.platform.reporting.figure_planning import GroupReport
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


type GroupDrawer = Callable[[GroupReport, GroupStyle, Path], Path | None]
"""Draws one group figure of an outcome into a path; ``None`` when the data cannot support it."""


def _paired_by_method(
    comparisons: Mapping[PreconditionerName, MethodComparison],
) -> dict[PreconditionerName, PairedComparison]:
    """The paired comparisons of the methods that had any system solved alongside the reference."""
    return {name: comparison.paired for name, comparison in comparisons.items()}


def _profile_only(
    draw: Callable[[RecipeProfileReport, GroupStyle, Path], Path | None],
) -> GroupDrawer:
    """Adapt a drawer of per-system outcomes; workload mixtures have none to draw."""

    def drawer(outcome: GroupReport, style: GroupStyle, path: Path) -> Path | None:
        match outcome:
            case RecipeProfileReport():
                return draw(outcome, style, path)
            case WorkloadMixtureReport():
                return None
            case unreachable:
                assert_never(unreachable)

    return drawer


def _time_bars(scenario: TimingScenario) -> GroupDrawer:
    def drawer(outcome: GroupReport, style: GroupStyle, path: Path) -> Path | None:
        return plot_online_time_bars(
            outcome.summaries, scenario=scenario.value, style=style, save_path=path
        )

    return drawer


def _performance_profile(cost: Cost) -> GroupDrawer:
    return _profile_only(
        lambda outcome, style, path: plot_performance_profile(
            outcome.outcomes, style=style, cost=cost, save_path=path
        )
    )


def _ecdf_time(cost: Cost, xlabel: str) -> GroupDrawer:
    return _profile_only(
        lambda outcome, style, path: plot_ecdf_profile(
            outcome.outcomes, style=style, cost=cost, xlabel=xlabel, save_path=path
        )
    )


def _aggregate_bars(outcome: GroupReport, style: GroupStyle, path: Path) -> Path | None:
    return plot_aggregate_bars(outcome, style=style, save_path=path)


def _paired_scatter(outcome: RecipeProfileReport, style: GroupStyle, path: Path) -> Path | None:
    reference = outcome.reference_selection
    if not isinstance(reference, ConfiguredReference):
        return None
    return plot_paired_time_scatter(
        outcome.outcomes, reference=reference, style=style, save_path=path
    )


def _speed_ratio(outcome: RecipeProfileReport, style: GroupStyle, path: Path) -> Path | None:
    reference = outcome.reference_selection
    return plot_speed_ratio_cdf(
        _paired_by_method(outcome.comparisons),
        style=style,
        reference_name=reference.name if isinstance(reference, ConfiguredReference) else None,
        save_path=path,
    )


GROUP_DRAWERS: Mapping[tuple[FigureKind, TimingScenario | None], GroupDrawer] = {
    (FigureKind.AGGREGATE_BARS, None): _aggregate_bars,
    (FigureKind.SOLVE_TIME, TimingScenario.SOLVE): _time_bars(TimingScenario.SOLVE),
    (FigureKind.ONLINE_TIME, TimingScenario.ONLINE): _time_bars(TimingScenario.ONLINE),
    (FigureKind.PAIRED_TIME, TimingScenario.ONLINE): _profile_only(_paired_scatter),
    (FigureKind.SPEED_RATIO, TimingScenario.ONLINE): _profile_only(_speed_ratio),
    (FigureKind.PERFORMANCE_PROFILE, TimingScenario.SOLVE): _performance_profile(solve_seconds),
    (FigureKind.PERFORMANCE_PROFILE, TimingScenario.ONLINE): _performance_profile(
        amortized_online_seconds
    ),
    (FigureKind.ECDF_ITERATIONS, None): _profile_only(
        lambda outcome, style, path: plot_ecdf_profile(
            outcome.outcomes, style=style, save_path=path
        )
    ),
    (FigureKind.ECDF_TIME, TimingScenario.SOLVE): _ecdf_time(
        solve_seconds, "Solve time per system [s]"
    ),
    (FigureKind.ECDF_TIME, TimingScenario.ONLINE): _ecdf_time(
        amortized_online_seconds, "Amortized online time per system [s]"
    ),
}
"""One drawer per group figure, keyed by the kind and timing scenario a plan records."""


def write_outcome_plots(
    outcome: GroupReport,
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan[GroupFigureMetadata]],
) -> tuple[Path, ...]:
    """Draw every planned figure the outcome's data supports, in plan order."""
    drawn = (
        GROUP_DRAWERS[plan.metadata.kind, plan.metadata.scenario](
            outcome, styler.style(plan), directory / plan.filename
        )
        for plan in plans
    )
    return tuple(path for path in drawn if path is not None)
