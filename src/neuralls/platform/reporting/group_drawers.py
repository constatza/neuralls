"""The drawer of each group figure kind: adapts a report group to one plot function.

A drawer receives the metric of the variant being drawn, so a kind with one variant per metric
(the time ECDF, the performance profile) needs a single drawer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import assert_never

from neuralls.domain.benchmarking.paired import (
    paired_systems,
    reference_among_methods,
    speed_ratio_summary,
    speed_ratios,
)
from neuralls.domain.benchmarking.profiles import ecdf_series, performance_series
from neuralls.domain.benchmarking.workload_totals import (
    WorkloadTotal,
    solve_total_seconds,
    workload_online_seconds,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_aggregate_bars,
    plot_ecdf_profile,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speed_ratio_cdf,
    plot_workload_bars,
)

type GroupDrawer = Callable[[MetricId, GroupReport, GroupStyle, Path], Path | None]
"""Draws the figure of one metric of a group report into a path; ``None`` when the data cannot
support it."""

type _ProfileDrawer = Callable[[MetricId, RecipeProfileReport, GroupStyle, Path], Path | None]

_ITERATIONS_XLABEL: Mapping[MetricId, str] = {MetricId.ITERATIONS: "Iterations"}
_TIME_XLABELS: Mapping[MetricId, str] = {
    MetricId.SOLVE_S: "Solve time per system [s]",
    MetricId.ONLINE_S: "Amortized online time per system [s]",
}


def _profile_only(draw: _ProfileDrawer) -> GroupDrawer:
    """Adapt a drawer that needs one homogeneous recipe; a workload mixture has none to draw."""

    def drawer(
        metric: MetricId, outcome: GroupReport, style: GroupStyle, path: Path
    ) -> Path | None:
        match outcome:
            case RecipeProfileReport():
                return draw(metric, outcome, style, path)
            case WorkloadMixtureReport():
                return None
            case unreachable:
                assert_never(unreachable)

    return drawer


def _time_bars(total: WorkloadTotal, xlabel: str) -> GroupDrawer:
    def drawer(
        metric: MetricId, outcome: GroupReport, style: GroupStyle, path: Path
    ) -> Path | None:
        return plot_workload_bars(
            outcome.summaries, total=total, xlabel=xlabel, style=style, save_path=path
        )

    return drawer


def _ecdf(xlabels: Mapping[MetricId, str]) -> GroupDrawer:
    return _profile_only(
        lambda metric, outcome, style, path: plot_ecdf_profile(
            ecdf_series(outcome.observations, metric),
            style=style,
            xlabel=xlabels[metric],
            save_path=path,
        )
    )


def _performance_profile(
    metric: MetricId, outcome: RecipeProfileReport, style: GroupStyle, path: Path
) -> Path | None:
    return plot_performance_profile(
        performance_series(outcome.observations, metric), style=style, save_path=path
    )


def _aggregate_bars(
    metric: MetricId, outcome: GroupReport, style: GroupStyle, path: Path
) -> Path | None:
    return plot_aggregate_bars(outcome, style=style, save_path=path)


def _paired_scatter(
    metric: MetricId, outcome: RecipeProfileReport, style: GroupStyle, path: Path
) -> Path | None:
    reference = reference_among_methods(outcome.observations, outcome.reference_selection)
    if reference is None:
        return None
    return plot_paired_time_scatter(
        paired_systems(outcome.observations, reference, metric),
        reference_name=reference,
        style=style,
        save_path=path,
    )


def _speed_ratio(
    metric: MetricId, outcome: RecipeProfileReport, style: GroupStyle, path: Path
) -> Path | None:
    reference = reference_among_methods(outcome.observations, outcome.reference_selection)
    if reference is None:
        return None
    paired = paired_systems(outcome.observations, reference, metric)
    return plot_speed_ratio_cdf(
        speed_ratios(paired),
        speed_ratio_summary(outcome.observations, reference, metric),
        style=style,
        reference_name=reference,
        save_path=path,
    )


AGGREGATE_BARS_DRAWER: GroupDrawer = _aggregate_bars
SOLVE_TIME_BARS_DRAWER: GroupDrawer = _time_bars(solve_total_seconds, "Solve time [s]")
ONLINE_TIME_BARS_DRAWER: GroupDrawer = _time_bars(
    workload_online_seconds, "Total online workload time [s]"
)
PAIRED_TIME_DRAWER: GroupDrawer = _profile_only(_paired_scatter)
SPEED_RATIO_DRAWER: GroupDrawer = _profile_only(_speed_ratio)
PERFORMANCE_PROFILE_DRAWER: GroupDrawer = _profile_only(_performance_profile)
ECDF_ITERATIONS_DRAWER: GroupDrawer = _ecdf(_ITERATIONS_XLABEL)
ECDF_TIME_DRAWER: GroupDrawer = _ecdf(_TIME_XLABELS)
