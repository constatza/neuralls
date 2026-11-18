"""The full-cost figure: each method's offline stages stacked on its pair workload sums.

Offline cost (dataset generation and the offline fit) is paid once per method artifact and online
cost (each pair's matrix setup and solve) once per workload, so a bar is the full cost of serving
the report's pairs from a fresh artifact. Colour encodes the stage and the method
is the x tick label, so one fixed colour means one stage in every bar. A stage with no recorded
number draws no segment and the bar says so instead of ending at a full cost that would read the
unknown stage as free. A method fit inside the comparison has its measured fit as its offline fit
segment, and the text above the bar says where the fit ran.
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
from neuralls.platform.reporting.figure_table import FullStatus
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.offline_cost_codec import FitOriginKind, StageStatus
from neuralls.platform.reporting.series_style import color_pool
from neuralls.platform.reporting.stacked_segments import segment_key

type CostRow = Mapping[str, object]

GENERATION_LABEL = "dataset generation (offline)"
FIT_LABEL = "offline fit"
SETUP_LABEL = "matrix setup, charged to each pair"
SOLVE_LABEL = "solves"
"""Words of the four stages in the legend; the unknown-full-cost note reuses the offline two."""

_OKABE_ITO_SLOTS = 8
"""Size of the colour-blind-safe palette the stage colours are picked from."""


class CostSegment(StrEnum):
    """One stacked part of a cost bar, named by the full-cost breakdown column it draws."""

    GENERATION = "generation_s"
    FIT = "fit_s"
    SETUP = "workload_setup_s"
    SOLVE = "workload_solve_s"


def cost_segments() -> tuple[CostSegment, ...]:
    """The segments stacked, bottom to top: offline stages first, then the online ones."""
    return (CostSegment.GENERATION, CostSegment.FIT, CostSegment.SETUP, CostSegment.SOLVE)


def _segment_label(segment: CostSegment) -> str:
    match segment:
        case CostSegment.GENERATION:
            return GENERATION_LABEL
        case CostSegment.FIT:
            return FIT_LABEL
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
        case CostSegment.FIT:
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
    """The full cost in seconds, or which offline stages leave it unknown."""
    match FullStatus(row["full_status"]):
        case FullStatus.COMPLETE:
            return f"{row['full_s']:.4g} s"
        case FullStatus.OFFLINE_UNAVAILABLE:
            unavailable = [
                label
                for label, status in (
                    ("generation", row["generation_status"]),
                    (FIT_LABEL, row["fit_status"]),
                )
                if status == StageStatus.UNAVAILABLE.value
            ]
            return f"full cost unknown: {' and '.join(unavailable)} unavailable"
        case unreachable:
            assert_never(unreachable)


def fit_origin_note(row: CostRow) -> tuple[str, ...]:
    """The line saying where the method was fit; none for a method with nothing to fit."""
    match FitOriginKind(row["fit_origin"]):
        case FitOriginKind.NONE:
            return ()
        case FitOriginKind.INLINE:
            return ("fit in comparison, measured",)
        case FitOriginKind.JOB:
            return ("fit job",)
        case unreachable:
            assert_never(unreachable)


def cost_bar_text(row: CostRow) -> str:
    """The text above a bar: the full cost (or why it is unknown), then the fit origin line."""
    return "\n".join((cost_bar_label(row), *fit_origin_note(row)))


def plot_full_cost_breakdown(
    table: pl.DataFrame, *, style: GroupStyle, target: FigureTarget
) -> FigureTarget | None:
    """Draw one stacked bar per method of the full-cost breakdown ``table``."""
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
    # The stage legend already names the offline and online parts; a longer label would run
    # past the axes' height into the subtitle.
    ax.set_ylabel("Full cost [s]")
    ax.set_xlabel("Method")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        show_titles=style.show_titles,
        target=target,
        show=False,
        saved_message="Saved full cost breakdown to",
    )
    return target


def _segment_key(segment: CostSegment) -> Patch:
    return segment_key(_segment_label(segment), 1.0, None, segment_color(segment))
