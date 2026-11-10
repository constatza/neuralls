"""Distributions of a per-system metric, and the per-matrix breakdown as a method-by-matrix heatmap.

Every summary mark is read from the figure's table. A distribution's box (q25, q50, q75), its
whiskers (q05 to q95, a distribution summary and not a confidence interval) and its "k/n"
annotation come from the method's cohort summary row; seaborn draws only the violin density of the
systems and the per-matrix mean points, which have no table counterpart and are not persisted. At
a mixture level each recipe gets a panel and the pooled rows stay in the table, as for the mean
bars (``seaborn_means.py``).

A heatmap's cells are the rows of its matrix breakdown: seaborn colours them from the pivoted
breakdown, and each cell's annotation, or the hatching and "0/r solved" of a cell where nothing
converged, is read from the same row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from neuralls.domain.solver.reporting.cohorts import COHORT_COLUMN, MIXTURE_COHORT
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.cohort_points import cohort_systems
from neuralls.platform.reporting.figure_annotations import (
    RowAnnotation,
    annotate_rows,
    count_label,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_style import rotate_x_labels
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.platform.reporting.seaborn_means import draw_recipe_panels
from neuralls.platform.reporting.series_style import MethodPalette, method_palette
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName

_MARK_COLOR = "black"
_WHISKER_CAP = 0.15
"""Half-height of a whisker cap, in units of the distance between two method rows."""
_BOX_LINE_WIDTH = 5.0
_MEDIAN_MARKER_SIZE = 5.0
_POINT_SIZE = 3.0
_POINT_ALPHA = 0.7
_POINT_JITTER = 0.15
_VIOLIN_ALPHA = 0.45

_CELL_WIDTH_IN = 1.1
"""Width in inches of one method's heatmap column."""
_CELL_HEIGHT_IN = 0.4
"""Height in inches of one matrix's heatmap row; the figure grows with the number of matrices."""
_HEATMAP_MARGIN_IN = 2.5
"""Room in inches for the title, the method labels below the cells and the colour bar."""
_MIN_HEATMAP_HEIGHT_IN = 4.0
_HATCH = "///"
_HATCH_COLOR = "0.6"
_CELL_FONT_SIZE = 7
_HEATMAP_CMAP = "viridis"
_LIGHT_CELL = 0.6
"""Normalized value above which a viridis cell is light enough for black text."""


@dataclass(frozen=True, slots=True)
class Quantiles:
    """The quantiles a violin's box and whiskers mark, and the largest value its density reaches."""

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
    """One method's row of a distribution panel and the units its quantiles rest on."""

    method: PreconditionerName
    quantiles: Quantiles | NoQuantiles
    n_defined: int
    n_planned: int


@dataclass(frozen=True, slots=True)
class DistributionPanel:
    """The distributions of one cohort, the recipe key that names it and its rows' population."""

    cohort: str
    population: Population
    methods: tuple[MethodDistribution, ...]


def distribution_panels(summary: pl.DataFrame) -> tuple[DistributionPanel, ...]:
    """One panel per recipe cohort of ``summary``, in table order; the mixture rows are not drawn."""
    drawn = summary.filter(pl.col(COHORT_COLUMN) != MIXTURE_COHORT)
    return tuple(
        _panel(cohort, drawn.filter(pl.col(COHORT_COLUMN) == cohort))
        for cohort in drawn[COHORT_COLUMN].unique(maintain_order=True).to_list()
    )


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


def distribution_xlabel(metric: MetricId) -> str:
    """The x label of a distribution of ``metric``."""
    return METRICS[metric].label.capitalize()


def plot_distribution(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Violins of each method's systems with the summary's box, whiskers and "k/n" counts.

    ``summary`` is the cohort summary table of the figure; ``None`` when it has no recipe cohort.
    """
    panels = distribution_panels(summary)
    palette = method_palette(
        sorted({method.method for panel in panels for method in panel.methods}), style
    )

    def draw_panel(ax: Axes, column: int) -> None:
        _draw_distribution_panel(ax, panels[column], observations, palette, metric)
        ax.set_xlabel(distribution_xlabel(metric))

    return draw_recipe_panels(
        [panel.cohort for panel in panels],
        palette,
        draw_panel=draw_panel,
        style=style,
        target=target,
        saved_message="Saved distribution to",
    )


def _draw_distribution_panel(
    ax: Axes,
    panel: DistributionPanel,
    observations: pl.DataFrame,
    palette: MethodPalette,
    metric: MetricId,
) -> None:
    labels, order = palette.labels, palette.order
    systems = cohort_systems(observations, panel, metric).select(
        pl.col("method").replace_strict(labels).alias("label"),
        value=pl.col(METRICS[metric].column).cast(pl.Float64),
    )
    points = matrix_points(observations, panel, metric)
    if systems.height > 0:
        sns.violinplot(
            data=plot_frame(systems),
            x="value",
            y="label",
            order=order,
            hue="label",
            hue_order=order,
            palette=dict(palette.colors),
            legend=False,
            inner=None,
            cut=0,
            density_norm="width",
            linewidth=0.5,
            alpha=_VIOLIN_ALPHA,
            orient="h",
            ax=ax,
        )
        sns.stripplot(
            data=plot_frame(
                points.select(pl.col("method").replace_strict(labels).alias("label"), "value")
            ),
            x="value",
            y="label",
            order=order,
            color=_MARK_COLOR,
            size=_POINT_SIZE,
            alpha=_POINT_ALPHA,
            jitter=_POINT_JITTER,
            orient="h",
            ax=ax,
        )
    rows = {label: position for position, label in enumerate(order)}
    point_ends = dict(points.group_by("method").agg(pl.col("value").max()).iter_rows())
    annotations = []
    for method in panel.methods:
        row = rows[labels[method.method]]
        start = max(_draw_box(ax, method.quantiles, row), point_ends.get(method.method, 0.0))
        annotations.append(
            RowAnnotation(row, start, count_label(method.n_defined, method.n_planned, metric))
        )
    annotate_rows(ax, annotations)


def _draw_box(ax: Axes, quantiles: Quantiles | NoQuantiles, row: int) -> float:
    """Draw the whiskers, box and median of ``quantiles`` on ``row``; where its density ends."""
    match quantiles:
        case Quantiles(q05=q05, q25=q25, q50=q50, q75=q75, q95=q95, maximum=maximum):
            ax.plot([q05, q95], [row, row], color=_MARK_COLOR, linewidth=1)
            for end in (q05, q95):
                ax.plot(
                    [end, end],
                    [row - _WHISKER_CAP, row + _WHISKER_CAP],
                    color=_MARK_COLOR,
                    linewidth=1,
                )
            ax.plot(
                [q25, q75],
                [row, row],
                color=_MARK_COLOR,
                linewidth=_BOX_LINE_WIDTH,
                solid_capstyle="butt",
            )
            ax.plot(
                [q50],
                [row],
                marker="o",
                markersize=_MEDIAN_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=_MARK_COLOR,
                linestyle="none",
            )
            return maximum
        case NoQuantiles():
            return 0.0
        case unreachable:
            assert_never(unreachable)


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
        figsize=(
            len(palette.order) * _CELL_WIDTH_IN + _HEATMAP_MARGIN_IN,
            max(_MIN_HEATMAP_HEIGHT_IN, len(matrices) * _CELL_HEIGHT_IN + _HEATMAP_MARGIN_IN),
        )
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=_HEATMAP_CMAP,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Mean over converged RHS: {METRICS[metric].label}"},
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
