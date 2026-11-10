"""Paired time scatter and speed-ratio CDF against the reference method.

A scatter pins a failed run to an axis edge as a hollow marker; the paired values and ratios are
frames from ``domain.benchmarking.paired``, which this module only draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from neuralls.domain.benchmarking.paired import PairKind
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import (
    paired_scatter_figure_size,
    with_outside_legend,
)
from neuralls.platform.reporting.figure_text import (
    LEGEND_ANCHOR,
    LEGEND_FONT_SIZE,
    draw_legend_outside,
)
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import method_palette
from neuralls.shared.types import PreconditionerName

FAILED_AXIS_MARGIN = 1.6
"""Factor above the slowest solved run at which failures are drawn, and below the fastest at
which the log axes start; the upper one is a position marking "did not converge", not a time."""


def _hollow(ax: Axes, xs: Sequence[float], ys: Sequence[float], *, color: Any, marker: Any) -> None:
    """Draw unfilled markers: the run behind each did not converge, so its value is an axis edge.

    The markers sit on the axes' limit, so they are not clipped to the axes or half of each would
    be cut off.
    """
    ax.scatter(xs, ys, facecolors="none", edgecolors=color, marker=marker, s=28, clip_on=False)


@dataclass(frozen=True, slots=True)
class PairedLogAxes:
    """The shared log range of both scatter axes and where a failed run is drawn on them."""

    low: float
    high: float
    edge: float


def paired_log_axes(times: Sequence[float]) -> PairedLogAxes | None:
    """Axes spanning the positive ``times`` with a margin, or ``None`` when there are none.

    A log axis cannot place a time of 0, so such a time neither sets the range nor is drawn.
    The upper end is the failure position, so a hollow marker sits on the axis edge.
    """
    positive = [time for time in times if time > 0]
    if not positive:
        return None
    high = max(positive) * FAILED_AXIS_MARGIN
    return PairedLogAxes(low=min(positive) / FAILED_AXIS_MARGIN, high=high, edge=high)


def plot_paired_time_scatter(
    paired: pl.DataFrame,
    *,
    reference_name: PreconditionerName,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Time of each method against the reference method's time, system by system, on log axes.

    ``paired`` is a ``paired_systems`` frame of ``metric``. The reference is the method explicitly
    selected in configuration. A point below the diagonal is a system on which the method beat
    the reference. Log axes keep systems that differ by orders of magnitude apart and put equal
    speed ratios at equal distances from the diagonal. Where only one of the two failed the point
    sits on the failing axis's upper edge as a hollow marker; where both failed there is nothing
    to compare and the system is not in the frame.
    """
    all_times = pl.concat([paired["reference_value"], paired["method_value"]]).drop_nulls()
    axes = paired_log_axes(all_times.to_list())
    if axes is None:
        return None
    others = paired["method"].unique().sort().to_list()
    palette = method_palette(others, style)
    label = METRICS[metric].label

    fig, ax = plt.subplots(
        figsize=paired_scatter_figure_size([style.legend.label(name) for name in others])
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([axes.low, axes.high], [axes.low, axes.high], color="gray", linewidth=1, linestyle="--")
    for name in others:
        pairs = paired.filter(pl.col("method") == name)
        both = pairs.filter(pl.col("pair_kind") == PairKind.BOTH.value)
        method_only = pairs.filter(pl.col("pair_kind") == PairKind.METHOD_ONLY.value)
        reference_only = pairs.filter(pl.col("pair_kind") == PairKind.REFERENCE_ONLY.value)
        color, marker = palette.color(name), palette.marker(name)
        ax.scatter(
            both["reference_value"].to_list(),
            both["method_value"].to_list(),
            color=color, marker=marker, s=28,
            label=style.legend.label(name),
        )  # fmt: skip
        _hollow(
            ax,
            [axes.edge] * method_only.height,
            method_only["method_value"].to_list(),
            color=color,
            marker=marker,
        )
        _hollow(
            ax,
            reference_only["reference_value"].to_list(),
            [axes.edge] * reference_only.height,
            color=color,
            marker=marker,
        )
    ax.set_xlim(axes.low, axes.high)
    ax.set_ylim(axes.low, axes.high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        f"{style.legend.label(reference_name)} (reference): {label}"
        "\nhollow markers: the other axis's run did not converge"
    )
    ax.set_ylabel(f"Method: {label}")
    ax.grid(True, which="both", alpha=0.3)
    draw_legend_outside(ax, style.legend)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved paired time scatter to",
    )
    return target


def speed_ratio_legend(
    label: str,
    *,
    median: float | None,
    ci: tuple[float, float] | None,
) -> str:
    """Legend text of a speed-ratio curve: the median and its interval.

    What the ratio divides is on the x label, not repeated here. The system and matrix counts the interval rests on are on the figure's card, not in the legend.
    """
    text = f"{label}: median {median:.2f}" if median is not None else f"{label}: no paired ratio"
    if ci is not None:
        text += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return text


def _speed_ratio_curve(ratios: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted reference/method ratios and their cumulative fractions."""
    ordered = np.sort(np.asarray(ratios, dtype=np.float64))
    fractions = np.arange(1, len(ordered) + 1) / len(ordered) if len(ordered) else np.asarray([])
    return ordered, fractions


def plot_speed_ratio_cdf(
    ratios: pl.DataFrame,
    summary: pl.DataFrame,
    *,
    style: GroupStyle,
    reference_name: PreconditionerName,
    target: FigureTarget,
) -> FigureTarget | None:
    """ECDF of per-system reference/method speed ratios.

    ``ratios`` holds the ``speed_ratios`` rows and ``summary`` the ``speed_ratio_summary`` rows
    of the same pairing, one per method. Only systems both methods solved enter. The median speed
    ratio carries a percentile-bootstrap interval (Efron & Tibshirani 1993) that resamples
    whole matrices, so sibling right-hand sides do not count as independent
    observations. Each legend entry shows the median and its interval. The vertical line at 1 separates systems where the method was slower
    (left) from faster (right).
    """
    if summary.height == 0:
        return None
    names = summary["method"].to_list()
    palette = method_palette(names, style)
    entries = [
        speed_ratio_legend(
            style.legend.label(row["method"]),
            median=row["median_ratio"],
            ci=None if row["ci_low"] is None else (row["ci_low"], row["ci_high"]),
        )
        for row in summary.iter_rows(named=True)
    ]
    fig, ax = plt.subplots(figsize=with_outside_legend((8.0, 5.0), entries))
    for row in summary.iter_rows(named=True):
        name = row["method"]
        xs, ys = _speed_ratio_curve(ratios.filter(pl.col("method") == name)["ratio"].to_list())
        ci = None if row["ci_low"] is None else (row["ci_low"], row["ci_high"])
        ax.step(
            xs,
            ys,
            where="post",
            color=palette.color(name),
            linestyle=palette.linestyle(name),
            label=speed_ratio_legend(
                style.legend.label(name),
                median=row["median_ratio"],
                ci=ci,
            ),
        )
    ax.axvline(1.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(
        f"Speed ratio, {style.legend.label(reference_name)} (reference) / method"
        "\n(slower < 1, faster > 1)"
    )
    ax.set_ylabel("Fraction of paired systems")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=LEGEND_ANCHOR, fontsize=LEGEND_FONT_SIZE)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved speed ratio CDF to",
    )
    return target
