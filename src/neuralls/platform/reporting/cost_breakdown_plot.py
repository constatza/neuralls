"""The total-cost figure: each method's offline stages stacked on its online workload totals.

Offline cost (dataset generation and training) is paid once per method artifact and online cost
(matrix setups and solves) once per workload, so a bar is the total cost of serving the report's
workload from a fresh artifact. A stage with no recorded number draws no segment and the bar says
so instead of ending at a total that would read the unknown stage as free.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import assert_never

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch

from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import bar_figure_size
from neuralls.platform.reporting.figure_table import TotalStatus
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.offline_cost_codec import StageStatus
from neuralls.platform.reporting.series_style import method_palette
from neuralls.platform.reporting.stacked_segments import segment_key

type CostRow = Mapping[str, object]


class CostSegment(StrEnum):
    """One stacked part of a cost bar, named by the cost-breakdown column it draws."""

    GENERATION = "generation_s"
    TRAINING = "training_s"
    SETUP = "allocated_setup_s"
    SOLVE = "solve_total_s"


def cost_segments() -> tuple[CostSegment, ...]:
    """The segments stacked, left to right: offline stages first, then the online ones."""
    return (CostSegment.GENERATION, CostSegment.TRAINING, CostSegment.SETUP, CostSegment.SOLVE)


def _segment_look(segment: CostSegment) -> tuple[str, float, str | None]:
    """Legend label, opacity of the method colour and hatch of ``segment``."""
    match segment:
        case CostSegment.GENERATION:
            return "dataset generation (offline)", 0.6, "xx"
        case CostSegment.TRAINING:
            return "training (offline)", 0.6, "++"
        case CostSegment.SETUP:
            return "matrix setup", 0.35, None
        case CostSegment.SOLVE:
            return "solves", 1.0, None
        case unreachable:
            assert_never(unreachable)


def cost_bar_label(row: CostRow) -> str:
    """The text ending a bar: the total in seconds, or which offline stages leave it unknown."""
    match TotalStatus(row["total_status"]):
        case TotalStatus.COMPLETE:
            return f"{row['total_s']:.4g} s"
        case TotalStatus.OFFLINE_UNAVAILABLE:
            unavailable = [
                stage
                for stage in ("generation", "training")
                if row[f"{stage}_status"] == StageStatus.UNAVAILABLE.value
            ]
            return f"total unknown: {' and '.join(unavailable)} unavailable"
        case unreachable:
            assert_never(unreachable)


def plot_cost_breakdown(
    table: pl.DataFrame, *, style: GroupStyle, target: FigureTarget
) -> FigureTarget | None:
    """Draw one stacked bar per method of the cost-breakdown ``table``."""
    if table.height == 0:
        return None
    rows = list(table.iter_rows(named=True))
    names = [row["method_id"] for row in rows]
    palette = method_palette(names, style)
    labels = [style.legend.label(name) for name in names]
    colors = [palette.color(name) for name in names]
    fig, ax = plt.subplots(figsize=bar_figure_size(labels))
    positions = np.arange(len(names))
    left = np.zeros(len(names))
    for segment in cost_segments():
        _, alpha, hatch = _segment_look(segment)
        widths = np.asarray([row[segment.value] or 0.0 for row in rows], dtype=np.float64)
        ax.barh(
            positions, widths, left=left, color=colors, alpha=alpha, hatch=hatch, edgecolor=colors
        )
        left = left + widths
    ax.set_yticks(positions, labels, fontsize=LEGEND_FONT_SIZE)
    for position, end, row in zip(positions, left, rows, strict=True):
        ax.text(end, position, f" {cost_bar_label(row)}", va="center")
    ax.legend(
        handles=[_segment_key(segment) for segment in cost_segments()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
        fontsize=LEGEND_FONT_SIZE,
    )
    ax.set_xlabel("Total seconds: offline (once per method) + setup + solve")
    ax.set_ylabel("Method")
    ax.grid(True, axis="x", alpha=0.3)
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
    label, alpha, hatch = _segment_look(segment)
    return segment_key(label, alpha, hatch)
