"""Aggregate convergence-count bars of a report group."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import bar_figure_size
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import method_palette


def plot_aggregate_bars(
    summary: pl.DataFrame,
    *,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Plot explicit convergence counts for every method in a report group.

    ``summary`` is the ``SUMMARY_SCHEMA`` table of the iteration metric, whose defined systems are
    the converged ones; a system was attempted unless it was never run or its setup failed.
    """
    if summary.height == 0:
        return None
    rows = list(summary.iter_rows(named=True))
    names = [row["method"] for row in rows]
    palette = method_palette(names, style)
    labels = [style.legend.label(name) for name in names]
    fig, ax = plt.subplots(figsize=bar_figure_size(labels))
    values = [row["n_defined"] for row in rows]
    positions = np.arange(len(names))
    bars = ax.barh(positions, values, color=[palette.color(name) for name in names])
    ax.set_yticks(positions, labels, fontsize=LEGEND_FONT_SIZE)
    ax.set_xlim(0, max(1, *(row["n_planned"] for row in rows)))
    ax.set_xlabel("Converged RHS (of planned RHS)")
    ax.set_ylabel("Method")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, row in zip(bars, rows, strict=True):
        n_attempted = row["n_defined"] + row["n_unconverged"] + row["n_solver_error"]
        bar_label = f"{row['n_defined']}/{n_attempted}/{row['n_planned']}"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {bar_label}", va="center")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved aggregate bars to",
    )
    return target
