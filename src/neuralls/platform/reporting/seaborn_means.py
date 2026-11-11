"""Bars of per-method means and of time per iteration, one figure per cohort.

Every summary mark is read from the figure's summary frame, the table stored beside it: the bar
is the frame's statistic column, the whisker its matrix-cluster interval columns and the "k/n"
annotation its ``n_defined`` and ``n_planned``. Seaborn draws the bars with ``errorbar=None``
because its own error bars resample rows independently, which is wrong when the RHS of one matrix
share it. Only the per-system points of time per iteration, which have no table counterpart, are
read from the observations.

A figure draws one cohort: the rows of a recipe, or of the pooled mixture, of the cohort table.
Planning gives every cohort its own figure (``cohort_variants.py``), so a figure has one axes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.domain.solver.reporting.cohorts import single_cohort
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.cohort_points import panel_points
from neuralls.platform.reporting.figure_annotations import (
    ColumnAnnotation,
    annotate_columns,
    count_label,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import Statistic
from neuralls.platform.reporting.figure_sizing import POINT_JITTER, bar_columns_figure_size
from neuralls.platform.reporting.figure_style import rotate_x_labels
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.mark_legend import SYSTEM_POINT_STYLE, Mark, draw_mark_legend
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.platform.reporting.series_style import MethodPalette, method_palette
from neuralls.shared.types import PreconditionerName

type BarStatistic = Literal[
    Statistic.MEAN_OVER_SYSTEMS, Statistic.MEAN_OVER_MATRICES, Statistic.RATIO_OF_SUMS
]
"""The statistics drawn as bars; each value names the summary column the bar is read from."""

_WHISKER_COLOR = "black"
_WHISKER_CAP = 0.15
"""Half-width of a whisker cap, in units of the distance between two method columns."""
_POINT_SIZE = 3.0
_NO_POINTS = pl.DataFrame(schema={"method": pl.String, "value": pl.Float64})
"""The points of a mean figure, whose bars stand alone."""


@dataclass(frozen=True, slots=True)
class Whisker:
    """The matrix-cluster bootstrap interval of a bar's statistic."""

    low: float
    high: float


@dataclass(frozen=True, slots=True)
class NoWhisker:
    """The bar has no interval: a ratio of sums, or fewer than two matrices to resample."""


@dataclass(frozen=True, slots=True)
class Bar:
    """A drawn bar: the statistic of a method that has defined values."""

    value: float
    whisker: Whisker | NoWhisker


@dataclass(frozen=True, slots=True)
class NoBar:
    """The method has no defined value, so only its count is drawn; a gap is never a zero."""


@dataclass(frozen=True, slots=True)
class MethodBar:
    """One method's column of a panel and the units its statistic rests on."""

    method: PreconditionerName
    height: Bar | NoBar
    n_defined: int
    n_planned: int


@dataclass(frozen=True, slots=True)
class BarPanel:
    """The bars of one cohort, the name that its table gives it, and the population of its rows."""

    cohort: str
    population: Population
    bars: tuple[MethodBar, ...]


def bar_panel(summary: pl.DataFrame, statistic: BarStatistic) -> BarPanel:
    """The panel of the one cohort ``summary`` holds, a recipe's or the pooled mixture's."""
    return _panel(single_cohort(summary), summary, statistic)


def _panel(cohort: str, rows: pl.DataFrame, statistic: BarStatistic) -> BarPanel:
    return BarPanel(
        cohort,
        Population(rows["population"][0]),
        tuple(
            MethodBar(
                PreconditionerName(row["method"]),
                _height(row, statistic),
                n_defined=row["n_defined"],
                n_planned=row["n_planned"],
            )
            for row in rows.iter_rows(named=True)
        ),
    )


def _height(row: Mapping[str, object], statistic: BarStatistic) -> Bar | NoBar:
    match row[statistic.value]:
        case float() as value:
            return Bar(value, _whisker(row, statistic))
        case _:
            return NoBar()


def _whisker(row: Mapping[str, object], statistic: BarStatistic) -> Whisker | NoWhisker:
    match statistic:
        case Statistic.MEAN_OVER_SYSTEMS | Statistic.MEAN_OVER_MATRICES:
            match (row[f"{statistic.value}_ci_low"], row[f"{statistic.value}_ci_high"]):
                case (float() as low, float() as high):
                    return Whisker(low, high)
                case _:
                    return NoWhisker()
        case Statistic.RATIO_OF_SUMS:
            return NoWhisker()
        case unreachable:
            assert_never(unreachable)


def defined_label(bar: MethodBar, metric: MetricId) -> str:
    """The annotation of ``bar``: how many of its planned units define ``metric``."""
    return count_label(bar.n_defined, bar.n_planned, metric)


def bar_statistic_text(statistic: BarStatistic) -> str:
    """How a bar statistic is named on its figure's axis and in its card's metric definition."""
    match statistic:
        case Statistic.MEAN_OVER_SYSTEMS:
            return "mean over systems"
        case Statistic.MEAN_OVER_MATRICES:
            return "mean over matrices"
        case Statistic.RATIO_OF_SUMS:
            return "ratio of sums"
        case unreachable:
            assert_never(unreachable)


def plot_mean_bars(
    summary: pl.DataFrame,
    *,
    statistic: BarStatistic,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Bars of each method's mean of ``metric`` with matrix-cluster whiskers and "k/n" counts.

    ``summary`` is the rows of one cohort of the figure's cohort summary table; ``None`` when it
    has none.
    """
    if summary.height == 0:
        return None
    return _draw_cohort(
        bar_panel(summary, statistic),
        _NO_POINTS,
        statistic=statistic,
        metric=metric,
        style=style,
        target=target,
        saved_message="Saved mean bars to",
    )


def plot_time_per_iteration(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Ratio-of-sums bars of time per iteration over each method's per-system rates.

    ``summary`` is the rows of one cohort of the cohort ratio-of-sums table; the points are the
    systems of ``observations`` (those of the same cohort) that returned an iterate, which no
    table stores.
    """
    if summary.height == 0:
        return None
    panel = bar_panel(summary, Statistic.RATIO_OF_SUMS)
    return _draw_cohort(
        panel,
        panel_points(observations, panel, metric),
        statistic=Statistic.RATIO_OF_SUMS,
        metric=metric,
        style=style,
        target=target,
        saved_message="Saved time per iteration bars to",
    )


def _draw_cohort(
    panel: BarPanel,
    points: pl.DataFrame,
    *,
    statistic: BarStatistic,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
    saved_message: str,
) -> FigureTarget:
    """Draw ``panel`` on one axes, titled with the scope and the cohort by ``style``."""
    palette = method_palette(sorted({bar.method for bar in panel.bars}), style)
    fig, ax = plt.subplots(figsize=bar_columns_figure_size(palette.order, style.label_rotation_deg))
    marks = _draw_panel(ax, panel, points, palette, metric)
    ax.set_ylabel(f"{bar_statistic_text(statistic).capitalize()}: {METRICS[metric].label}")
    ax.set_xlabel("Method")
    rotate_x_labels(ax, style.label_rotation_deg)
    draw_mark_legend(ax, marks | {_bar_mark(statistic)})
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        show_titles=style.show_titles,
        target=target,
        show=False,
        saved_message=saved_message,
    )
    return target


def _draw_panel(
    ax: Axes,
    panel: BarPanel,
    points: pl.DataFrame,
    palette: MethodPalette,
    metric: MetricId,
) -> frozenset[Mark]:
    """Draw the bars, whiskers and points of ``panel``; the marks other than the bar drawn."""
    labels = palette.labels
    order = palette.order
    bars = pl.DataFrame(
        {
            "label": [labels[bar.method] for bar in panel.bars],
            "value": [_value(bar) for bar in panel.bars],
        },
        schema={"label": pl.String, "value": pl.Float64},
    )
    sns.barplot(
        data=plot_frame(bars),
        x="label",
        y="value",
        order=order,
        hue="label",
        hue_order=order,
        palette=dict(palette.colors),
        legend=False,
        errorbar=None,
        orient="v",
        ax=ax,
    )
    if points.height > 0:
        sns.stripplot(
            data=plot_frame(
                points.select(pl.col("method").replace_strict(labels).alias("label"), "value")
            ),
            x="label",
            y="value",
            order=order,
            size=_POINT_SIZE,
            jitter=POINT_JITTER,
            orient="v",
            ax=ax,
            **SYSTEM_POINT_STYLE,
        )
    columns = {label: position for position, label in enumerate(order)}
    # The annotation stands on the method's highest mark, so it never covers a point or whisker.
    point_tops = dict(points.group_by("method").agg(pl.col("value").max()).iter_rows())
    annotations = []
    for bar in panel.bars:
        column = columns[labels[bar.method]]
        base = max(_draw_whisker(ax, bar, column), point_tops.get(bar.method, 0.0))
        annotations.append(ColumnAnnotation(column, base, defined_label(bar, metric)))
    annotate_columns(ax, annotations)
    return frozenset(
        mark
        for mark, drawn in (
            (Mark.CONFIDENCE_INTERVAL, any(_has_whisker(bar) for bar in panel.bars)),
            (Mark.SYSTEM_POINTS, points.height > 0),
        )
        if drawn
    )


def _bar_mark(statistic: BarStatistic) -> Mark:
    match statistic:
        case Statistic.MEAN_OVER_SYSTEMS | Statistic.MEAN_OVER_MATRICES:
            return Mark.MEAN_BAR
        case Statistic.RATIO_OF_SUMS:
            return Mark.RATIO_BAR
        case unreachable:
            assert_never(unreachable)


def _has_whisker(bar: MethodBar) -> bool:
    match bar.height:
        case Bar(whisker=Whisker()):
            return True
        case Bar() | NoBar():
            return False
        case unreachable:
            assert_never(unreachable)


def _value(bar: MethodBar) -> float | None:
    match bar.height:
        case Bar(value=value):
            return value
        case NoBar():
            return None
        case unreachable:
            assert_never(unreachable)


def _draw_whisker(ax: Axes, bar: MethodBar, column: int) -> float:
    """Draw the whisker of ``bar`` on ``column``; the y where its annotation stands."""
    match bar.height:
        case Bar(value=value, whisker=whisker):
            return max(value, _whisker_top(ax, whisker, column))
        case NoBar():
            return 0.0
        case unreachable:
            assert_never(unreachable)


def _whisker_top(ax: Axes, whisker: Whisker | NoWhisker, column: int) -> float:
    """Draw ``whisker`` on ``column``; its upper end, or 0 when there is none."""
    match whisker:
        case Whisker(low=low, high=high):
            ax.plot([column, column], [low, high], color=_WHISKER_COLOR, linewidth=1)
            for end in (low, high):
                ax.plot(
                    [column - _WHISKER_CAP, column + _WHISKER_CAP],
                    [end, end],
                    color=_WHISKER_COLOR,
                    linewidth=1,
                )
            return high
        case NoWhisker():
            return 0.0
        case unreachable:
            assert_never(unreachable)
