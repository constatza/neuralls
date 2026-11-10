"""The drawer of each group figure kind: adapts a figure's data to one plot function.

A drawer receives the ``FigureData`` of one variant, so a kind with one variant per metric (the
time ECDF, the performance profile) needs a single drawer. The ``summary`` of the data is the
frame stored beside the figure; a drawer draws that frame and reads nothing else of the report
except the observations that place what the frame cannot (the denominators of methods that solved
nothing, the systems paired against the reference).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import polars as pl

from neuralls.domain.benchmarking.paired import (
    paired_systems,
    reference_among_methods,
    speed_ratios,
)
from neuralls.domain.benchmarking.profiles import ProfileSeries, ecdf_series, performance_series
from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.platform.reporting.cost_breakdown_plot import plot_cost_breakdown
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import Statistic
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.paired_plots import plot_paired_time_scatter, plot_speed_ratio_cdf
from neuralls.platform.reporting.profile_plots import plot_ecdf_profile, plot_performance_profile
from neuralls.platform.reporting.seaborn_cost_curves import (
    plot_break_even_curves,
    plot_convergence_factors,
)
from neuralls.platform.reporting.seaborn_distributions import (
    plot_distribution,
    plot_matrix_heatmap,
)
from neuralls.platform.reporting.seaborn_means import plot_mean_bars, plot_time_per_iteration
from neuralls.platform.reporting.workload_bars import WorkloadTotal, plot_workload_bars


@dataclass(frozen=True, slots=True)
class FigureData:
    """What one group figure is drawn from.

    Attributes:
        observations: The observation view of the report group the figure belongs to.
        summary: The frame the figure draws and its sibling table stores; its schema is the
            schema of the kind of table the figure's recipe writes.
        style: Title, scope, legend and series styling.
        metric: The per-system metric the figure summarizes.
        reference: The reference method the group's configuration selected, for the figures that
            compare methods against it.
    """

    observations: pl.DataFrame
    summary: pl.DataFrame
    style: GroupStyle
    metric: MetricId
    reference: ReferenceSelection


type GroupDrawer = Callable[[FigureData, FigureTarget], FigureTarget | None]
"""Draws one figure to a target, which it returns; ``None`` when the data cannot support it."""

type _SeriesOf = Callable[[pl.DataFrame, MetricId], ProfileSeries]

_ITERATIONS_XLABEL: Mapping[MetricId, str] = {MetricId.ITERATIONS: "Iterations"}
_TIME_XLABELS: Mapping[MetricId, str] = {
    metric: METRICS[metric].label.capitalize() for metric in (MetricId.SOLVE_S, MetricId.ONLINE_S)
}


def _profile_series(data: FigureData, series_of: _SeriesOf) -> ProfileSeries:
    """The series to draw: the curves are the summary, the denominators come from the systems.

    A method that solved nothing has no curve point, so its planned count, like the number of
    systems a performance profile excluded, is only found among the observations.
    """
    return replace(series_of(data.observations, data.metric), frame=data.summary)


def solve_time_bars(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_workload_bars(
        data.summary,
        total=WorkloadTotal.SOLVE,
        ylabel="Solve time [s]",
        style=data.style,
        target=target,
    )


def online_time_bars(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_workload_bars(
        data.summary,
        total=WorkloadTotal.ONLINE,
        ylabel="Total online workload time [s]",
        style=data.style,
        target=target,
    )


def ecdf_iterations(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_ecdf_profile(
        _profile_series(data, ecdf_series),
        style=data.style,
        xlabel=_ITERATIONS_XLABEL[data.metric],
        target=target,
    )


def ecdf_time(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_ecdf_profile(
        _profile_series(data, ecdf_series),
        style=data.style,
        xlabel=_TIME_XLABELS[data.metric],
        target=target,
    )


def performance_profile(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_performance_profile(
        _profile_series(data, performance_series), style=data.style, target=target
    )


def cost_breakdown(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_cost_breakdown(data.summary, style=data.style, target=target)


def paired_time_scatter(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    match reference_among_methods(data.observations, data.reference):
        case None:
            return None
        case reference:
            return plot_paired_time_scatter(
                data.summary,
                reference_name=reference,
                metric=data.metric,
                style=data.style,
                target=target,
            )


def speed_ratio_cdf(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    match reference_among_methods(data.observations, data.reference):
        case None:
            return None
        case reference:
            paired = paired_systems(data.observations, reference, data.metric)
            return plot_speed_ratio_cdf(
                speed_ratios(paired),
                data.summary,
                style=data.style,
                reference_name=reference,
                target=target,
            )


def break_even(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    match reference_among_methods(data.observations, data.reference):
        case None:
            return None
        case reference:
            return plot_break_even_curves(
                data.summary, reference_name=reference, style=data.style, target=target
            )


def convergence_factors(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_convergence_factors(
        data.summary, data.observations, metric=data.metric, style=data.style, target=target
    )


def mean_over_systems_bars(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_mean_bars(
        data.summary,
        statistic=Statistic.MEAN_OVER_SYSTEMS,
        metric=data.metric,
        style=data.style,
        target=target,
    )


def mean_over_matrices_bars(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_mean_bars(
        data.summary,
        statistic=Statistic.MEAN_OVER_MATRICES,
        metric=data.metric,
        style=data.style,
        target=target,
    )


def time_per_iteration(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_time_per_iteration(
        data.summary, data.observations, metric=data.metric, style=data.style, target=target
    )


def distribution(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_distribution(
        data.summary, data.observations, metric=data.metric, style=data.style, target=target
    )


def matrix_heatmap(data: FigureData, target: FigureTarget) -> FigureTarget | None:
    return plot_matrix_heatmap(data.summary, metric=data.metric, style=data.style, target=target)
