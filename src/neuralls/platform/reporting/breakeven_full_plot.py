"""The offline break-even figure: N* of each ordered method pair, with its interval.

One row per method with a known offline cost and method it is compared against: a dot at the pair
count N* from which its offline cost is paid off, a whisker over the finite part of its two-level
bootstrap interval, and the numbers written beside it, since an interval reaching "never" (an
infinite upper bound) has no end to draw.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import assert_never

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from neuralls.domain.benchmarking.cost_curves import BreakEvenState
from neuralls.platform.reporting.figure_annotations import RowAnnotation, annotate_rows
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import method_rows_figure_size
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import method_palette

type BreakEvenRow = Mapping[str, object]


def _bound(value: object) -> str:
    return "inf" if value == math.inf else f"{value:.3g}"


def breakeven_full_text(row: BreakEvenRow) -> str:
    """N* in pairs, never, or no common pairs; then the 95% interval when there is one."""
    match BreakEvenState(row["break_even"]):
        case BreakEvenState.BREAKS_EVEN:
            text = f"N* = {row['breakeven_full']:.3g} pairs"
        case BreakEvenState.NEVER:
            text = "never breaks even (not faster per pair)"
        case BreakEvenState.NO_COMMON_SYSTEMS:
            return "no pair both converged"
        case unreachable:
            assert_never(unreachable)
    if row["ci_low"] is None:
        return text
    return f"{text}, 95% CI [{_bound(row['ci_low'])}, {_bound(row['ci_high'])}]"


def _finite_marks(row: BreakEvenRow) -> list[float]:
    values = (row["breakeven_full"], row["ci_low"], row["ci_high"])
    return [value for value in values if isinstance(value, float) and math.isfinite(value)]


def plot_breakeven_full(
    table: pl.DataFrame, *, style: GroupStyle, target: FigureTarget
) -> FigureTarget | None:
    """Draw one row per ordered method pair of the ``breakeven_full`` ``table``."""
    if table.height == 0:
        return None
    rows = list(table.iter_rows(named=True))
    palette = method_palette(sorted({row["method"] for row in rows}), style)
    labels = [
        f"{style.legend.label(row['method'])}\nvs {style.legend.label(row['reference_method'])}"
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=method_rows_figure_size(labels))
    annotations: list[RowAnnotation] = []
    for position, row in zip(np.arange(len(rows)), rows, strict=True):
        marks = _finite_marks(row)
        color = palette.color(row["method"])
        if len(marks) > 1:
            ax.hlines(position, min(marks), max(marks), color=color)
        if row["breakeven_full"] is not None:
            ax.plot(
                row["breakeven_full"], position, marker=palette.marker(row["method"]), color=color
            )
        annotations.append(
            RowAnnotation(int(position), max(marks, default=0.0), breakeven_full_text(row))
        )
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=LEGEND_FONT_SIZE)
    ax.invert_yaxis()
    annotate_rows(ax, annotations)
    ax.set_xlabel("Offline break-even N* [pairs]")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        show_titles=style.show_titles,
        target=target,
        show=False,
        saved_message="Saved offline break-even to",
    )
    return target
