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
from pathlib import Path

import polars as pl

from neuralls.domain.benchmarking.paired import (
    paired_systems,
    reference_among_methods,
    speed_ratios,
)
from neuralls.domain.benchmarking.profiles import ProfileSeries, ecdf_series, performance_series
from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_metadata import Statistic
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    TotalColumn,
    plot_aggregate_bars,
    plot_ecdf_profile,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speed_ratio_cdf,
    plot_workload_bars,
)
from neuralls.platform.reporting.seaborn_distributions import (
    plot_distribution,
    plot_matrix_heatmap,
)
from neuralls.platform.reporting.seaborn_means import (
    BarStatistic,
    plot_mean_bars,
    plot_time_per_iteration,
)


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


type GroupDrawer = Callable[[FigureData, Path], Path | None]
"""Draws one figure into a path; ``None`` when the data cannot support it."""

type _SeriesOf = Callable[[pl.DataFrame, MetricId], ProfileSeries]

_ITERATIONS_XLABEL: Mapping[MetricId, str] = {MetricId.ITERATIONS: "Iterations"}
_TIME_XLABELS: Mapping[MetricId, str] = {
    MetricId.SOLVE_S: "Solve time per system [s]",
    MetricId.ONLINE_S: "Amortized online time per system [s]",
}


def _profile_series(data: FigureData, series_of: _SeriesOf) -> ProfileSeries:
    """The series to draw: the curves are the summary, the denominators come from the systems.

    A method that solved nothing has no curve point, so its planned count, like the number of
    systems a performance profile excluded, is only found among the observations.
    """
    return replace(series_of(data.observations, data.metric), frame=data.summary)


def _time_bars(column: TotalColumn, xlabel: str) -> GroupDrawer:
    def drawer(data: FigureData, path: Path) -> Path | None:
        return plot_workload_bars(
            data.summary, column=column, xlabel=xlabel, style=data.style, save_path=path
        )

    return drawer


def _ecdf(xlabels: Mapping[MetricId, str]) -> GroupDrawer:
    def drawer(data: FigureData, path: Path) -> Path | None:
        return plot_ecdf_profile(
            _profile_series(data, ecdf_series),
            style=data.style,
            xlabel=xlabels[data.metric],
            save_path=path,
        )

    return drawer


def _performance_profile(data: FigureData, path: Path) -> Path | None:
    return plot_performance_profile(
        _profile_series(data, performance_series), style=data.style, save_path=path
    )


def _aggregate_bars(data: FigureData, path: Path) -> Path | None:
    return plot_aggregate_bars(data.summary, style=data.style, save_path=path)


def _paired_scatter(data: FigureData, path: Path) -> Path | None:
    match reference_among_methods(data.observations, data.reference):
        case None:
            return None
        case reference:
            return plot_paired_time_scatter(
                data.summary, reference_name=reference, style=data.style, save_path=path
            )


def _speed_ratio(data: FigureData, path: Path) -> Path | None:
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
                save_path=path,
            )


def _mean_bars(statistic: BarStatistic) -> GroupDrawer:
    def drawer(data: FigureData, path: Path) -> Path | None:
        return plot_mean_bars(
            data.summary, statistic=statistic, metric=data.metric, style=data.style, save_path=path
        )

    return drawer


def _time_per_iteration(data: FigureData, path: Path) -> Path | None:
    return plot_time_per_iteration(
        data.summary, data.observations, metric=data.metric, style=data.style, save_path=path
    )


def _distribution(data: FigureData, path: Path) -> Path | None:
    return plot_distribution(
        data.summary, data.observations, metric=data.metric, style=data.style, save_path=path
    )


def _matrix_heatmap(data: FigureData, path: Path) -> Path | None:
    return plot_matrix_heatmap(data.summary, metric=data.metric, style=data.style, save_path=path)


AGGREGATE_BARS_DRAWER: GroupDrawer = _aggregate_bars
SOLVE_TIME_BARS_DRAWER: GroupDrawer = _time_bars("solve_total_s", "Solve time [s]")
ONLINE_TIME_BARS_DRAWER: GroupDrawer = _time_bars(
    "workload_online_s", "Total online workload time [s]"
)
PAIRED_TIME_DRAWER: GroupDrawer = _paired_scatter
SPEED_RATIO_DRAWER: GroupDrawer = _speed_ratio
PERFORMANCE_PROFILE_DRAWER: GroupDrawer = _performance_profile
ECDF_ITERATIONS_DRAWER: GroupDrawer = _ecdf(_ITERATIONS_XLABEL)
ECDF_TIME_DRAWER: GroupDrawer = _ecdf(_TIME_XLABELS)
MEAN_OVER_SYSTEMS_DRAWER: GroupDrawer = _mean_bars(Statistic.MEAN_OVER_SYSTEMS)
MEAN_OVER_MATRICES_DRAWER: GroupDrawer = _mean_bars(Statistic.MEAN_OVER_MATRICES)
TIME_PER_ITERATION_DRAWER: GroupDrawer = _time_per_iteration
DISTRIBUTION_DRAWER: GroupDrawer = _distribution
MATRIX_HEATMAP_DRAWER: GroupDrawer = _matrix_heatmap
