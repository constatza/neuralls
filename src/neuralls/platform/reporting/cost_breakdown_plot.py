"""The total-cost figure: each method's offline stages stacked on its online workload totals.

Offline cost (dataset generation and the offline fit or training) is paid once per method
artifact and online cost (matrix setups and solves) once per workload, so a bar is the total cost
of serving the report's workload from a fresh artifact. Colour encodes the stage and the method
is the x tick label, so one fixed colour means one stage in every bar. A stage with no recorded
number draws no segment and the bar says so instead of ending at a total that would read the
unknown stage as free; a method fit inline has no offline fit segment because that fit is part of
its setup segment, which the text above the bar says.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import assert_never

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch

from neuralls.platform.reporting.figure_annotations import ColumnAnnotation, annotate_columns
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import bar_columns_figure_size
from neuralls.platform.reporting.figure_style import rotate_x_labels
from neuralls.platform.reporting.figure_table import TotalStatus
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.offline_cost_codec import FitOriginKind, StageStatus
from neuralls.platform.reporting.series_style import color_pool
from neuralls.platform.reporting.stacked_segments import segment_key

type CostRow = Mapping[str, object]

GENERATION_LABEL = "dataset generation (offline)"
TRAINING_LABEL = "offline fit / training"
SETUP_LABEL = "matrix setup"
SOLVE_LABEL = "solves"
"""Words of the four stages in the legend; the unknown-total note reuses the offline two."""

_OKABE_ITO_SLOTS = 8
"""Size of the colour-blind-safe palette the stage colours are picked from."""


class CostSegment(StrEnum):
    """One stacked part of a cost bar, named by the cost-breakdown column it draws."""

    GENERATION = "generation_s"
    TRAINING = "training_s"
    SETUP = "allocated_setup_s"
    SOLVE = "solve_total_s"


def cost_segments() -> tuple[CostSegment, ...]:
    """The segments stacked, bottom to top: offline stages first, then the online ones."""
    return (CostSegment.GENERATION, CostSegment.TRAINING, CostSegment.SETUP, CostSegment.SOLVE)


def _segment_label(segment: CostSegment) -> str:
    match segment:
        case CostSegment.GENERATION:
            return GENERATION_LABEL
        case CostSegment.TRAINING:
            return TRAINING_LABEL
        case CostSegment.SETUP:
            return SETUP_LABEL
        case CostSegment.SOLVE:
            return SOLVE_LABEL
        case unreachable:
            assert_never(unreachable)


def _palette_slot(segment: CostSegment) -> int:
    """The Okabe-Ito slot of ``segment``: sky blue, orange, green, blue, which stay apart."""
    match segment:
        case CostSegment.GENERATION:
            return 1
        case CostSegment.TRAINING:
            return 0
        case CostSegment.SETUP:
            return 2
        case CostSegment.SOLVE:
            return 4
        case unreachable:
            assert_never(unreachable)


def segment_color(segment: CostSegment) -> str:
    """The one colour ``segment`` has in every bar and in the legend."""
    return str(color_pool(_OKABE_ITO_SLOTS)[_palette_slot(segment)])


def cost_bar_label(row: CostRow) -> str:
    """The total in seconds, or which offline stages leave it unknown."""
    match TotalStatus(row["total_status"]):
        case TotalStatus.COMPLETE:
            return f"{row['total_s']:.4g} s"
        case TotalStatus.OFFLINE_UNAVAILABLE:
            unavailable = [
                label
                for label, status in (
                    ("generation", row["generation_status"]),
                    (TRAINING_LABEL, row["training_status"]),
                )
                if status == StageStatus.UNAVAILABLE.value
            ]
            return f"total unknown: {' and '.join(unavailable)} unavailable"
        case unreachable:
            assert_never(unreachable)


def fit_origin_note(row: CostRow) -> tuple[str, ...]:
    """The line saying where the method was fit; none for a method with nothing to fit."""
    match FitOriginKind(row["fit_origin"]):
        case FitOriginKind.NONE:
            return ()
        case FitOriginKind.INLINE:
            return ("fit inline, counted in setup",)
        case FitOriginKind.JOB:
            return ("fit job",)
        case unreachable:
            assert_never(unreachable)


def cost_bar_text(row: CostRow) -> str:
    """The text above a bar: the total (or why it is unknown), then the fit origin line."""
    return "\n".join((cost_bar_label(row), *fit_origin_note(row)))


def plot_cost_breakdown(
    table: pl.DataFrame, *, style: GroupStyle, target: FigureTarget
) -> FigureTarget | None:
    """Draw one stacked bar per method of the cost-breakdown ``table``."""
    if table.height == 0:
        return None
    rows = list(table.iter_rows(named=True))
    names = [row["method_id"] for row in rows]
    labels = [style.legend.label(name) for name in names]
    fig, ax = plt.subplots(figsize=bar_columns_figure_size(labels, style.label_rotation_deg))
    positions = np.arange(len(names))
    bottom = np.zeros(len(names))
    for segment in cost_segments():
        heights = np.asarray([row[segment.value] or 0.0 for row in rows], dtype=np.float64)
        ax.bar(
            positions,
            heights,
            bottom=bottom,
            color=segment_color(segment),
        )
        bottom = bottom + heights
    ax.set_xticks(positions, labels, fontsize=LEGEND_FONT_SIZE)
    rotate_x_labels(ax, style.label_rotation_deg)
    annotate_columns(
        ax,
        [
            ColumnAnnotation(int(position), float(top), cost_bar_text(row))
            for position, top, row in zip(positions, bottom, rows, strict=True)
        ],
    )
    # Outside the axes on the right: below them it would sit on the rotated method labels.
    ax.legend(
        handles=[_segment_key(segment) for segment in cost_segments()],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
    )
    ax.set_ylabel("Total seconds: offline (once per method) + setup + solve")
    ax.set_xlabel("Method")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved cost breakdown to",
    )
    return target


def _segment_key(segment: CostSegment) -> Patch:
    return segment_key(_segment_label(segment), 1.0, None, segment_color(segment))
