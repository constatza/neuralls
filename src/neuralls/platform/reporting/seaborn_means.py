"""Bars of per-method means and of time per iteration, one panel per recipe.

Every summary mark is read from the figure's summary frame, the table stored beside it: the bar
is the frame's statistic column, the whisker its matrix-cluster interval columns and the "k/n"
annotation its ``n_defined`` and ``n_planned``. Seaborn draws the bars with ``errorbar=None``
because its own error bars resample rows independently, which is wrong when the RHS of one matrix
share it. Only the per-system points of time per iteration, which have no table counterpart, are
read from the observations.

At a mixture level the frame holds one cohort per recipe and a pooled ``MIXTURE_COHORT`` block;
each recipe gets a panel and the pooled rows stay in the table, since pooling hides how the
recipes differ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.domain.presentation.recipes import render_recipe
from neuralls.domain.solver.reporting.cohorts import COHORT_COLUMN, MIXTURE_COHORT
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.platform.reporting.figure_metadata import Statistic
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE, TITLE_PROFILE
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.platform.reporting.plots import finish_group_figure, group_series_styles
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.shared.rhs_recipe import parse_recipe_key
from neuralls.shared.types import PreconditionerName

type BarStatistic = Literal[
    Statistic.MEAN_OVER_SYSTEMS, Statistic.MEAN_OVER_MATRICES, Statistic.RATIO_OF_SUMS
]
"""The statistics drawn as bars; each value names the summary column the bar is read from."""

_PANEL_WIDTH_IN = 4.5
"""Width in inches of one recipe panel's bars."""
_LABEL_CHAR_WIDTH_IN = 0.055
"""Width in inches of one legend-size character, reserved for the method labels."""
_ROW_HEIGHT_IN = 0.45
"""Height in inches of one method's row."""
_MIN_HEIGHT_IN = 4.0
_HEADER_RATIO = 0.02
"""Height of the header row that carries the scope above all panels, relative to a panel."""
_WHISKER_COLOR = "black"
_WHISKER_CAP = 0.15
"""Half-height of a whisker cap, in units of the distance between two method rows."""
_ANNOTATION_ROOM = 1.45
"""Factor past the rightmost annotation start the x axis extends to, so "k/n" text fits inside."""
_POINT_SIZE = 3.0
_POINT_ALPHA = 0.6
_POINT_JITTER = 0.2
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
    """One method's row of a panel and the units its statistic rests on."""

    method: PreconditionerName
    height: Bar | NoBar
    n_defined: int
    n_planned: int


@dataclass(frozen=True, slots=True)
class BarPanel:
    """The bars of one cohort, the recipe key that names it, and the population of its rows."""

    cohort: str
    population: Population
    bars: tuple[MethodBar, ...]


def bar_panels(summary: pl.DataFrame, statistic: BarStatistic) -> tuple[BarPanel, ...]:
    """One panel per recipe cohort of ``summary``, in table order; the mixture rows are not drawn."""
    drawn = summary.filter(pl.col(COHORT_COLUMN) != MIXTURE_COHORT)
    return tuple(
        _panel(cohort, drawn.filter(pl.col(COHORT_COLUMN) == cohort), statistic)
        for cohort in drawn[COHORT_COLUMN].unique(maintain_order=True).to_list()
    )


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


def panel_points(observations: pl.DataFrame, panel: BarPanel, metric: MetricId) -> pl.DataFrame:
    """The per-system values behind ``panel``: its recipe's selected systems, ``method, value``."""
    return (
        select(observations, metric, panel.population)
        .filter(pl.col("recipe") == panel.cohort)
        .select("method", value=METRICS[metric].column)
    )


def defined_label(bar: MethodBar, metric: MetricId) -> str:
    """The annotation of ``bar``: how many of its planned units define ``metric``."""
    return f"{bar.n_defined}/{bar.n_planned} {_defined_word(metric)}"


def _defined_word(metric: MetricId) -> str:
    match metric:
        case MetricId.ITERATIONS | MetricId.SOLVE_S | MetricId.ONLINE_S | MetricId.STANDALONE_S:
            return "solved"
        case MetricId.SETUP_S:
            return "set up"
        case MetricId.TIME_PER_ITERATION_S:
            return "returned an iterate"
        case MetricId.CONVERGENCE_FACTOR:
            return "converged with a residual history"
        case unreachable:
            assert_never(unreachable)


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
    save_path: Path,
) -> Path | None:
    """Bars of each method's mean of ``metric`` with matrix-cluster whiskers and "k/n" counts.

    ``summary`` is the cohort summary table of the figure; ``None`` when it has no recipe cohort.
    """
    return _draw_panels(
        bar_panels(summary, statistic),
        {},
        statistic=statistic,
        metric=metric,
        style=style,
        save_path=save_path,
        saved_message="Saved mean bars to",
    )


def plot_time_per_iteration(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Ratio-of-sums bars of time per iteration over each method's per-system rates.

    ``summary`` is the cohort ratio-of-sums table; the points are the systems of ``observations``
    that returned an iterate, which no table stores.
    """
    panels = bar_panels(summary, Statistic.RATIO_OF_SUMS)
    return _draw_panels(
        panels,
        {panel.cohort: panel_points(observations, panel, metric) for panel in panels},
        statistic=Statistic.RATIO_OF_SUMS,
        metric=metric,
        style=style,
        save_path=save_path,
        saved_message="Saved time per iteration bars to",
    )


def _draw_panels(
    panels: Sequence[BarPanel],
    points: Mapping[str, pl.DataFrame],
    *,
    statistic: BarStatistic,
    metric: MetricId,
    style: GroupStyle,
    save_path: Path,
    saved_message: str,
) -> Path | None:
    """Draw ``panels`` side by side under one header that carries the scope."""
    if not panels:
        return None
    names = sorted({bar.method for panel in panels for bar in panel.bars})
    labels = {name: style.legend.label(name) for name in names}
    styles = group_series_styles(names, style.families, style.color_keys, style.marker_keys)
    palette = {labels[name]: styles[name]["color"] for name in names}
    widest = max(len(line) for label in labels.values() for line in label.split("\n"))
    fig = plt.figure(
        figsize=(
            widest * _LABEL_CHAR_WIDTH_IN + _PANEL_WIDTH_IN * len(panels),
            max(_MIN_HEIGHT_IN, len(names) * _ROW_HEIGHT_IN + 2.0),
        )
    )
    grid = fig.add_gridspec(2, len(panels), height_ratios=(_HEADER_RATIO, 1.0))
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    first = fig.add_subplot(grid[1, 0])
    axes = [
        first,
        *(fig.add_subplot(grid[1, column], sharey=first) for column in range(1, len(panels))),
    ]
    xlabel = f"{bar_statistic_text(statistic).capitalize()}: {METRICS[metric].label}"
    for column, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        _draw_panel(ax, panel, points.get(panel.cohort, _NO_POINTS), labels, palette, metric)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Method" if column == 0 else "")
        if column > 0:
            ax.tick_params(labelleft=False)
        if len(panels) > 1:
            recipe = parse_recipe_key(panel.cohort)
            ax.set_title(render_recipe(recipe, TITLE_PROFILE, style.notation), fontsize=9)
    finish_group_figure(
        fig,
        header,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message=saved_message,
    )
    return save_path


def _draw_panel(
    ax: Axes,
    panel: BarPanel,
    points: pl.DataFrame,
    labels: Mapping[PreconditionerName, str],
    palette: Mapping[str, object],
    metric: MetricId,
) -> None:
    order = list(labels.values())
    bars = pl.DataFrame(
        {
            "label": [labels[bar.method] for bar in panel.bars],
            "value": [_value(bar) for bar in panel.bars],
        },
        schema={"label": pl.String, "value": pl.Float64},
    )
    sns.barplot(
        data=plot_frame(bars),
        x="value",
        y="label",
        order=order,
        hue="label",
        hue_order=order,
        palette=dict(palette),
        legend=False,
        errorbar=None,
        orient="h",
        ax=ax,
    )
    if points.height > 0:
        sns.stripplot(
            data=plot_frame(
                points.select(pl.col("method").replace_strict(labels).alias("label"), "value")
            ),
            x="value",
            y="label",
            order=order,
            color=_WHISKER_COLOR,
            size=_POINT_SIZE,
            alpha=_POINT_ALPHA,
            jitter=_POINT_JITTER,
            orient="h",
            ax=ax,
        )
    rows = {label: position for position, label in enumerate(order)}
    # The annotation starts right of the method's last point, so it never covers a point.
    point_ends = dict(points.group_by("method").agg(pl.col("value").max()).iter_rows())
    starts = []
    for bar in panel.bars:
        row = rows[labels[bar.method]]
        start = max(_draw_whisker(ax, bar, row), point_ends.get(bar.method, 0.0))
        starts.append(start)
        ax.text(
            start, row, f" {defined_label(bar, metric)}", va="center", fontsize=LEGEND_FONT_SIZE
        )
    rightmost = max(starts, default=0.0)
    if rightmost > 0.0:
        ax.set_xlim(0.0, rightmost * _ANNOTATION_ROOM)
    else:
        ax.set_xlim(left=0.0)
    ax.grid(True, axis="x", alpha=0.3)


def _value(bar: MethodBar) -> float | None:
    match bar.height:
        case Bar(value=value):
            return value
        case NoBar():
            return None
        case unreachable:
            assert_never(unreachable)


def _draw_whisker(ax: Axes, bar: MethodBar, row: int) -> float:
    """Draw the whisker of ``bar`` on ``row``; the x where its annotation starts."""
    match bar.height:
        case Bar(value=value, whisker=whisker):
            return max(value, _whisker_end(ax, whisker, row))
        case NoBar():
            return 0.0
        case unreachable:
            assert_never(unreachable)


def _whisker_end(ax: Axes, whisker: Whisker | NoWhisker, row: int) -> float:
    """Draw ``whisker`` on ``row``; its right end, or 0 when there is none."""
    match whisker:
        case Whisker(low=low, high=high):
            ax.plot([low, high], [row, row], color=_WHISKER_COLOR, linewidth=1)
            for end in (low, high):
                ax.plot(
                    [end, end],
                    [row - _WHISKER_CAP, row + _WHISKER_CAP],
                    color=_WHISKER_COLOR,
                    linewidth=1,
                )
            return high
        case NoWhisker():
            return 0.0
        case unreachable:
            assert_never(unreachable)
