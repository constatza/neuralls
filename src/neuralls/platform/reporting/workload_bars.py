"""Workload-total bars: all matrix setups plus all solves per method, stacked by outcome.

The bars show totals, not a per-system share, stacked so the seconds spent on attempts that
failed are not read as cheap converged solves.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import method_palette
from neuralls.platform.reporting.stacked_segments import segment_key


class WorkloadTotal(StrEnum):
    """The workload total a bar figure draws, named by its workload-totals column."""

    SOLVE = "solve_total_s"
    ONLINE = "workload_online_s"


class BarSegment(StrEnum):
    """One stacked part of a workload bar, named by the workload-totals column it draws."""

    SETUP = "allocated_setup_s"
    CONVERGED = "converged_solve_s"
    FAILED = "failed_solve_s"


def workload_segments(total: WorkloadTotal) -> tuple[BarSegment, ...]:
    """The columns stacked, bottom to top, into a bar of ``total``; they sum to it.

    Setup is its own segment: one setup event serves every RHS of a matrix, converged or not,
    so it belongs to neither solve outcome.
    """
    match total:
        case WorkloadTotal.SOLVE:
            return (BarSegment.CONVERGED, BarSegment.FAILED)
        case WorkloadTotal.ONLINE:
            return (BarSegment.SETUP, BarSegment.CONVERGED, BarSegment.FAILED)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class _SegmentLook:
    """How a segment is filled: the method colour at ``alpha``, hatched when ``hatch`` is set."""

    label: str
    alpha: float
    hatch: str | None


def _segment_look(segment: BarSegment) -> _SegmentLook:
    match segment:
        case BarSegment.SETUP:
            return _SegmentLook("matrix setup", 0.35, None)
        case BarSegment.CONVERGED:
            return _SegmentLook("solves that converged", 1.0, None)
        case BarSegment.FAILED:
            return _SegmentLook("attempts that failed", 0.25, "///")
        case unreachable:
            assert_never(unreachable)


def workload_bar_label(*, n_converged: int, n_attempted: int, n_planned: int) -> str:
    """The count above a workload bar: converged of planned, and attempted when short of it."""
    converged = f"{n_converged}/{n_planned} converged"
    if n_attempted < n_planned:
        return f"{converged}, {n_attempted}/{n_planned} attempted"
    return converged


def plot_workload_bars(
    totals: pl.DataFrame,
    *,
    total: WorkloadTotal,
    ylabel: str,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Draw one workload total per method, stacked by ``workload_segments``.

    ``totals`` is the workload-totals table and every segment is one of its columns. A total is
    not a per-system amortized share. The seconds of failed attempts are a hatched segment and
    the count above each bar says how many planned RHS converged, so a method that fails fast
    is not read as a cheap one.
    """
    if totals.height == 0:
        return None
    rows = list(totals.iter_rows(named=True))
    names = [row["method_id"] for row in rows]
    palette = method_palette(names, style)
    labels = [style.legend.label(name) for name in names]
    colors = [palette.color(name) for name in names]
    fig, ax = plt.subplots(figsize=bar_columns_figure_size(labels, style.label_rotation_deg))
    positions = np.arange(len(names))
    bottom = np.zeros(len(names))
    segments = workload_segments(total)
    for segment in segments:
        look = _segment_look(segment)
        heights = np.asarray([row[segment.value] for row in rows], dtype=np.float64)
        ax.bar(
            positions,
            heights,
            bottom=bottom,
            color=colors,
            alpha=look.alpha,
            hatch=look.hatch,
            edgecolor=colors,
        )
        bottom = bottom + heights
    ax.set_xticks(positions, labels, fontsize=LEGEND_FONT_SIZE)
    rotate_x_labels(ax, style.label_rotation_deg)
    annotate_columns(
        ax,
        [
            ColumnAnnotation(
                int(position),
                float(top),
                workload_bar_label(
                    n_converged=row["n_converged"],
                    n_attempted=row["n_attempted_rhs"],
                    n_planned=row["n_planned_rhs"],
                ),
            )
            for position, top, row in zip(positions, bottom, rows, strict=True)
        ],
    )
    # Outside the axes on the right: below them it would sit on the rotated method labels.
    ax.legend(
        handles=[_key(segment) for segment in segments],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved workload bars to",
    )
    return target


def _key(segment: BarSegment) -> Patch:
    look = _segment_look(segment)
    return segment_key(look.label, look.alpha, look.hatch)
