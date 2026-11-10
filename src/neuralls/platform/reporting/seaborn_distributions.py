"""Distributions of a per-system metric, and the per-matrix breakdown as a method-by-matrix heatmap.

A distribution figure has one column per method, methods on the x axis and the metric on the y
axis. How a column is drawn depends on how many systems it holds (``distribution_mode.row_mode``):
the systems as a beeswarm with a median tick, a box over the beeswarm, or a violin with a box
inside it. The "k/n" annotation above a column and the median tick come from the method's row of
the figure's cohort summary table. Seaborn's box recomputes the quantiles from the systems with
linear interpolation, which is the interpolation of the table, so the two agree (a test pins it).
A whisker ends on the most extreme system inside q05 to q95 (matplotlib clips it to a data point), a distribution summary and not a confidence interval. The violin density,
the beeswarm and the per-matrix mean points have no table counterpart and are not persisted. At a
mixture level each cohort has its own figure, as for the mean bars (``seaborn_means.py``).

A heatmap's cells are the rows of its matrix breakdown: seaborn colours them from the pivoted
breakdown, and each cell's annotation, or the hatching and "0/r solved" of a cell where nothing
converged, is read from the same row.
"""

from __future__ import annotations

import textwrap
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from neuralls.domain.solver.reporting.cohorts import single_cohort
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.cohort_points import cohort_systems
from neuralls.platform.reporting.distribution_mode import (
    BoxAndSwarm,
    PointsOnly,
    RowLimits,
    RowMode,
    Violin,
    row_mode,
)
from neuralls.platform.reporting.figure_annotations import (
    ColumnAnnotation,
    annotate_columns,
    count_label,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import bar_columns_figure_size, heatmap_figure_size
from neuralls.platform.reporting.figure_style import rotate_x_labels
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.mark_legend import SYSTEM_POINT_STYLE, Mark, draw_mark_legend
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.platform.reporting.series_style import MethodPalette, method_palette
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName

_MARK_COLOR = "black"
_BOX_WIDTH = 0.4
_BOX_WHISKER_PERCENTILES = (5, 95)
"""Percentiles of the box's whiskers; the table's ``q05`` and ``q95``."""
_MEDIAN_TICK_HALF_WIDTH = 0.2
"""Half-width of a median tick, in units of the distance between two method columns."""
_MEDIAN_TICK_WIDTH = 1.5
_POINT_SIZE = 3.0
_POINT_ALPHA = 0.7
_POINT_JITTER = 0.15
_VIOLIN_ALPHA = 0.45

_COLORBAR_LABEL_WRAP = 22
"""Characters per line of the colour bar label, so it is not taller than the bar."""
_HATCH = "///"
_HATCH_COLOR = "0.6"
_CELL_FONT_SIZE = 7
_HEATMAP_CMAP = "viridis"
_LIGHT_CELL = 0.6
"""Normalized value above which a viridis cell is light enough for black text."""


@dataclass(frozen=True, slots=True)
class Quantiles:
    """The quantiles a method's median tick marks and the largest system, where its column ends."""

    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class NoQuantiles:
    """The method has no defined value, so only its count is drawn."""


@dataclass(frozen=True, slots=True)
class MethodDistribution:
    """One method's column of a distribution panel and the units its quantiles rest on."""

    method: PreconditionerName
    quantiles: Quantiles | NoQuantiles
    n_defined: int
    n_planned: int


@dataclass(frozen=True, slots=True)
class DistributionPanel:
    """The distributions of one cohort, the name its table gives it and its rows' population."""

    cohort: str
    population: Population
    methods: tuple[MethodDistribution, ...]


def distribution_panel(summary: pl.DataFrame) -> DistributionPanel:
    """The panel of the one cohort ``summary`` holds, a recipe's or the pooled mixture's."""
    return _panel(single_cohort(summary), summary)


def _panel(cohort: str, rows: pl.DataFrame) -> DistributionPanel:
    return DistributionPanel(
        cohort,
        Population(rows["population"][0]),
        tuple(
            MethodDistribution(
                PreconditionerName(row["method"]),
                _quantiles(row),
                n_defined=row["n_defined"],
                n_planned=row["n_planned"],
            )
            for row in rows.iter_rows(named=True)
        ),
    )


def _quantiles(row: Mapping[str, object]) -> Quantiles | NoQuantiles:
    match (row["q05"], row["q25"], row["q50"], row["q75"], row["q95"], row["max"]):
        case (
            float() as q05,
            float() as q25,
            float() as q50,
            float() as q75,
            float() as q95,
            float() as top,
        ):
            return Quantiles(q05, q25, q50, q75, q95, top)
        case _:
            return NoQuantiles()


def matrix_points(
    observations: pl.DataFrame, panel: DistributionPanel, metric: MetricId
) -> pl.DataFrame:
    """One point per method and matrix of ``panel``: the mean over its systems, ``method, value``.

    The points show how the matrices spread inside each violin, since the RHS of one matrix are
    not independent of each other.
    """
    return (
        cohort_systems(observations, panel, metric)
        .group_by("method", "dataset", "matrix_id", maintain_order=True)
        .agg(value=pl.col(METRICS[metric].column).cast(pl.Float64).mean())
        .select("method", "value")
    )


def distribution_ylabel(metric: MetricId) -> str:
    """The y label of a distribution of ``metric``."""
    return METRICS[metric].label.capitalize()


def plot_distribution(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """One column per method: points, a box over points, or a violin, by its number of systems.

    ``summary`` is the rows of one cohort of the figure's cohort summary table and
    ``observations`` the systems of that cohort; ``None`` when ``summary`` has no row.
    """
    if summary.height == 0:
        return None
    panel = distribution_panel(summary)
    palette = method_palette(sorted(method.method for method in panel.methods), style)
    fig, ax = plt.subplots(figsize=bar_columns_figure_size(palette.order, style.label_rotation_deg))
    marks = _draw_distribution_panel(ax, panel, observations, palette, metric, style.row_limits)
    ax.set_ylabel(distribution_ylabel(metric))
    ax.set_xlabel("Method")
    rotate_x_labels(ax, style.label_rotation_deg)
    draw_mark_legend(ax, marks)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved distribution to",
    )
    return target


def _draw_distribution_panel(
    ax: Axes,
    panel: DistributionPanel,
    observations: pl.DataFrame,
    palette: MethodPalette,
    metric: MetricId,
    limits: RowLimits,
) -> frozenset[Mark]:
    """Draw every column of ``panel`` and return the marks that were drawn."""
    labels, order = palette.labels, palette.order
    modes = {labels[m.method]: row_mode(m.n_defined, limits) for m in panel.methods}
    systems = cohort_systems(observations, panel, metric).select(
        pl.col("method").replace_strict(labels).alias("label"),
        value=pl.col(METRICS[metric].column).cast(pl.Float64),
    )
    swarmed = _labels_in(modes, PointsOnly, BoxAndSwarm)
    boxed = _labels_in(modes, BoxAndSwarm, Violin)
    violins = _labels_in(modes, Violin)
    if systems.height > 0:
        _draw_violins(ax, systems, violins, palette)
        _draw_swarm(ax, systems, swarmed, order)
        _draw_boxes(ax, systems, boxed, order)
        _draw_matrix_points(ax, matrix_points(observations, panel, metric), violins, palette)
    _draw_median_ticks(ax, panel, modes, palette)
    marks = frozenset(mark for label in set(systems["label"]) for mark in _marks_of(modes[label]))
    columns = {label: position for position, label in enumerate(order)}
    ax.set_xticks(range(len(order)), order)
    annotations = [
        ColumnAnnotation(
            columns[labels[method.method]],
            _column_top(method.quantiles),
            count_label(method.n_defined, method.n_planned, metric),
        )
        for method in panel.methods
    ]
    annotate_columns(ax, annotations)
    return marks


def _labels_in(modes: Mapping[str, RowMode], *kinds: type[RowMode]) -> frozenset[str]:
    """The labels whose mode is an instance of one of ``kinds``."""
    return frozenset(label for label, mode in modes.items() if isinstance(mode, kinds))


def _marks_of(mode: RowMode) -> frozenset[Mark]:
    """The marks a column in ``mode`` shows, for the legend."""
    match mode:
        case PointsOnly():
            return frozenset({Mark.MEDIAN, Mark.SYSTEM_POINTS})
        case BoxAndSwarm():
            return frozenset({Mark.IQR, Mark.MEDIAN, Mark.QUANTILE_WHISKER, Mark.SYSTEM_POINTS})
        case Violin():
            return frozenset(
                {
                    Mark.VIOLIN,
                    Mark.IQR,
                    Mark.MEDIAN,
                    Mark.QUANTILE_WHISKER,
                    Mark.MATRIX_MEAN_POINTS,
                }
            )
        case unreachable:
            assert_never(unreachable)


def _column_top(quantiles: Quantiles | NoQuantiles) -> float:
    """Where a column's annotation stands: the largest system, or the axis floor when none."""
    match quantiles:
        case Quantiles(maximum=maximum):
            return maximum
        case NoQuantiles():
            return 0.0
        case unreachable:
            assert_never(unreachable)


def _only(systems: pl.DataFrame, labels: Collection[str]) -> pl.DataFrame:
    """The systems of the methods in ``labels``; the others keep their column but get no mark."""
    return systems.filter(pl.col("label").is_in(list(labels)))


def _draw_violins(
    ax: Axes, systems: pl.DataFrame, labels: Collection[str], palette: MethodPalette
) -> None:
    if not labels:
        return
    sns.violinplot(
        data=plot_frame(_only(systems, labels)),
        x="label",
        y="value",
        order=palette.order,
        hue="label",
        hue_order=palette.order,
        palette=dict(palette.colors),
        legend=False,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.5,
        alpha=_VIOLIN_ALPHA,
        orient="v",
        ax=ax,
    )


def _draw_swarm(ax: Axes, systems: pl.DataFrame, labels: Collection[str], order: list[str]) -> None:
    """A beeswarm of the systems of ``labels``: every system visible, none jittered at random."""
    if not labels:
        return
    sns.swarmplot(
        data=plot_frame(_only(systems, labels)),
        x="label",
        y="value",
        order=order,
        size=_POINT_SIZE,
        orient="v",
        ax=ax,
        **SYSTEM_POINT_STYLE,
    )


def _draw_boxes(ax: Axes, systems: pl.DataFrame, labels: Collection[str], order: list[str]) -> None:
    """Boxes of the systems of ``labels``: IQR box, median line, whiskers q05 to q95, no fliers."""
    if not labels:
        return
    sns.boxplot(
        data=plot_frame(_only(systems, labels)),
        x="label",
        y="value",
        order=order,
        whis=_BOX_WHISKER_PERCENTILES,
        showfliers=False,
        fill=False,
        color=_MARK_COLOR,
        width=_BOX_WIDTH,
        linewidth=1.0,
        orient="v",
        ax=ax,
    )


def _draw_matrix_points(
    ax: Axes, points: pl.DataFrame, labels: Collection[str], palette: MethodPalette
) -> None:
    """Strip the per-matrix means over the violins of ``labels``."""
    drawn = _only(
        points.select(pl.col("method").replace_strict(palette.labels).alias("label"), "value"),
        labels,
    )
    if drawn.height == 0:
        return
    sns.stripplot(
        data=plot_frame(drawn),
        x="label",
        y="value",
        order=palette.order,
        color=_MARK_COLOR,
        size=_POINT_SIZE,
        alpha=_POINT_ALPHA,
        jitter=_POINT_JITTER,
        orient="v",
        ax=ax,
    )


def _draw_median_ticks(
    ax: Axes,
    panel: DistributionPanel,
    modes: Mapping[str, RowMode],
    palette: MethodPalette,
) -> None:
    """A short horizontal tick at the summary's median of each points-only column."""
    columns = {label: position for position, label in enumerate(palette.order)}
    for method in panel.methods:
        label = palette.labels[method.method]
        match method.quantiles:
            case Quantiles(q50=median) if isinstance(modes[label], PointsOnly):
                ax.hlines(
                    median,
                    columns[label] - _MEDIAN_TICK_HALF_WIDTH,
                    columns[label] + _MEDIAN_TICK_HALF_WIDTH,
                    color=_MARK_COLOR,
                    linewidth=_MEDIAN_TICK_WIDTH,
                )
            case _:
                continue


@dataclass(frozen=True, slots=True)
class CellMean:
    """The mean of the metric over the RHS of the matrix on which it is defined."""

    value: float


@dataclass(frozen=True, slots=True)
class NoneDefined:
    """No RHS of the matrix defines the metric for the method: the cell is hatched, never 0."""


@dataclass(frozen=True, slots=True)
class HeatmapCell:
    """One method's cell on one matrix and the RHS counts it rests on."""

    method: PreconditionerName
    dataset: DatasetId
    matrix_id: MatrixId
    value: CellMean | NoneDefined
    n_defined: int
    n_planned: int


def heatmap_cells(breakdown: pl.DataFrame) -> tuple[HeatmapCell, ...]:
    """The cells of ``breakdown``, one per row and in table order."""
    return tuple(
        HeatmapCell(
            PreconditionerName(row["method"]),
            DatasetId(row["dataset"]),
            MatrixId(row["matrix_id"]),
            _cell_value(row["mean"]),
            n_defined=row["n_defined"],
            n_planned=row["n_planned"],
        )
        for row in breakdown.iter_rows(named=True)
    )


def _cell_value(mean: object) -> CellMean | NoneDefined:
    match mean:
        case float() as value:
            return CellMean(value)
        case _:
            return NoneDefined()


def cell_label(cell: HeatmapCell, metric: MetricId) -> str:
    """The cell's mean, with "k/r" below it when only some RHS define it, or "0/r solved"."""
    match cell.value:
        case CellMean(value=value):
            mean = f"{value:.3g}"
            if cell.n_defined == cell.n_planned:
                return mean
            return f"{mean}\n{cell.n_defined}/{cell.n_planned}"
        case NoneDefined():
            return count_label(cell.n_defined, cell.n_planned, metric)
        case unreachable:
            assert_never(unreachable)


def matrix_label(matrix_id: MatrixId) -> str:
    """How a matrix row is named; the dataset is the figure's scope, so only the index varies."""
    return f"matrix {int(matrix_id)}"


def plot_matrix_heatmap(
    breakdown: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Method-by-matrix heatmap of the per-matrix mean of ``metric``, rows the matrices.

    ``breakdown`` is the matrix breakdown table of the figure; ``None`` when it has no cell.
    """
    cells = heatmap_cells(breakdown)
    if not cells:
        return None
    palette = method_palette(sorted({cell.method for cell in cells}), style)
    matrices = sorted({cell.matrix_id for cell in cells})
    rows = {matrix: position for position, matrix in enumerate(matrices)}
    columns = {label: position for position, label in enumerate(palette.order)}
    pivot = (
        plot_frame(
            breakdown.select(
                pl.col("matrix_id"),
                pl.col("method").replace_strict(palette.labels).alias("label"),
                "mean",
            ).pivot(on="label", index="matrix_id", values="mean")
        )
        .set_index("matrix_id")
        .reindex(index=[int(matrix) for matrix in matrices], columns=palette.order)
    )
    pivot.index = [matrix_label(matrix) for matrix in matrices]
    means = [mean for cell in cells for mean in _defined_mean(cell)]
    norm = Normalize(min(means, default=0.0), max(means, default=1.0))
    fig, ax = plt.subplots(
        figsize=heatmap_figure_size(palette.order, style.label_rotation_deg, len(matrices))
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=_HEATMAP_CMAP,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": textwrap.fill(
                f"Mean over converged RHS: {METRICS[metric].label}", _COLORBAR_LABEL_WRAP
            )
        },
    )
    for cell in cells:
        _draw_cell(
            ax, cell, rows[cell.matrix_id], columns[palette.labels[cell.method]], norm, metric
        )
    ax.set_xlabel("Method")
    ax.set_ylabel("Matrix")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", labelsize=LEGEND_FONT_SIZE)
    rotate_x_labels(ax, style.label_rotation_deg)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved matrix heatmap to",
    )
    return target


def _defined_mean(cell: HeatmapCell) -> tuple[float, ...]:
    match cell.value:
        case CellMean(value=value):
            return (value,)
        case NoneDefined():
            return ()
        case unreachable:
            assert_never(unreachable)


def _draw_cell(
    ax: Axes, cell: HeatmapCell, row: int, column: int, norm: Normalize, metric: MetricId
) -> None:
    """Annotate ``cell``, hatching it first when nothing converged on its matrix."""
    match cell.value:
        case CellMean(value=value):
            color = _MARK_COLOR if norm(value) > _LIGHT_CELL else "white"
        case NoneDefined():
            ax.add_patch(
                Rectangle(
                    (column, row), 1, 1, fill=False, hatch=_HATCH, edgecolor=_HATCH_COLOR, lw=0
                )
            )
            color = _MARK_COLOR
        case unreachable:
            assert_never(unreachable)
    ax.text(
        column + 0.5,
        row + 0.5,
        cell_label(cell, metric),
        ha="center",
        va="center",
        fontsize=_CELL_FONT_SIZE,
        color=color,
    )
