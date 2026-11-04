"""Group-level plots that keep failed runs visible instead of averaging them away.

A failed run (iteration cap or breakdown) has no time-to-solution, so none of
these plots gives it a value: profiles leave it unsolved (Dolan & Moré 2002,
Math. Program. 91; Hansen et al. 2016, arXiv:1605.03560) and the scatters pin
it to an axis edge as a hollow marker. Every plot function takes the path to
save to and returns it, or `None` when there was nothing to draw. The curves
and points themselves are frames from `domain.benchmarking.profiles` and
`domain.benchmarking.paired`; this module only draws them.

Profiles and scatters use amortized online seconds: solve plus the planned share of the
matrix setup event. The time bars show workload online totals: all matrix setups plus all
solves, not a per-system share.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from neuralls.domain.benchmarking.paired import PairKind
from neuralls.domain.benchmarking.profiles import ProfileSeries
from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.platform.reporting.figure_text import (
    LEGEND_FONT_SIZE,
    Legend,
    ScopeSubtitle,
    draw_legend,
)
from neuralls.platform.reporting.plots import finish_group_figure, group_series_styles
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName

_FAILED_AXIS_MARGIN = 1.6
"""Factor above the slowest solved run at which failures are drawn; a position on the
axis marking "did not converge", not a time."""


@dataclass(frozen=True, slots=True)
class GroupStyle:
    """How a group's preconditioners are titled and drawn, shared by every plot here.

    ``title`` is fixed editorial wording; the subtitle is the typed ``scope`` rendered
    when a figure is drawn, and every series is named by ``legend``.
    """

    title: str
    scope: FigureScope
    notation: NotationPolicy
    legend: Legend
    families: Mapping[str, Hashable]
    color_keys: Mapping[str, Hashable]
    marker_keys: Mapping[str, Hashable]

    @property
    def subtitle(self) -> str:
        """The scope under the title, rendered for this style's notation."""
        return ScopeSubtitle(self.scope, self.notation).text


_BAR_AREA_WIDTH_IN = 6.0
"""Width in inches kept for the bars themselves, however long the method labels are."""

_LABEL_CHAR_WIDTH_IN = 0.055
"""Width in inches of one legend-size character, used to reserve room for method labels."""

_LABEL_LINE_HEIGHT_IN = 0.3
"""Height in inches one wrapped label line takes on a bar chart's method axis."""


def _bar_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size that leaves the bars their own width and gives every label line its own row."""
    lines = [line for label in labels for line in label.split("\n")]
    widest = max((len(line) for line in lines), default=0)
    return (
        _BAR_AREA_WIDTH_IN + widest * _LABEL_CHAR_WIDTH_IN,
        max(5.0, len(lines) * _LABEL_LINE_HEIGHT_IN + 2.0),
    )


def plot_aggregate_bars(
    summary: pl.DataFrame,
    *,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Plot explicit convergence counts for every method in a report group.

    ``summary`` is the ``SUMMARY_SCHEMA`` table of the iteration metric, whose defined systems are
    the converged ones; a system was attempted unless it was never run or its setup failed.
    """
    if summary.height == 0:
        return None
    rows = list(summary.iter_rows(named=True))
    names = [row["method"] for row in rows]
    styles = _series_styles(names, style)
    labels = [style.legend.label(name) for name in names]
    fig, ax = plt.subplots(figsize=_bar_figure_size(labels))
    values = [row["n_defined"] for row in rows]
    positions = np.arange(len(names))
    bars = ax.barh(positions, values, color=[styles[name]["color"] for name in names])
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
        save_path=save_path,
        show=False,
        saved_message="Saved aggregate bars to",
    )
    return save_path


def _series_styles(names: Sequence[str], style: GroupStyle) -> dict[str, dict[str, Any]]:
    return group_series_styles(names, style.families, style.color_keys, style.marker_keys)


def _legend_outside(ax: Axes, legend: Legend) -> None:
    """Place the legend right of the axes so it never covers the points."""
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        ncols=legend.layout.columns,
    )


def _hollow(ax: Axes, xs: Sequence[float], ys: Sequence[float], *, color: Any, marker: Any) -> None:
    """Draw unfilled markers: the run behind each did not converge, so its value is an axis edge."""
    ax.scatter(xs, ys, facecolors="none", edgecolors=color, marker=marker, s=28)


def profile_legend(label: str, *, n_solved: int, n_planned: int) -> str:
    """Legend text of a profile curve: the method and how many of its planned systems it solved."""
    return f"{label} ({n_solved}/{n_planned} solved)"


def excluded_zero_best_note(n_excluded: int) -> str | None:
    """Legend note on systems a performance profile left out, or ``None`` when there are none.

    A system whose best value is 0 has no ratio for any method, so it is dropped from every
    curve; the note says how many, so the lower plateau is not read as failures.
    """
    if n_excluded == 0:
        return None
    return f"{_counted(n_excluded, 'system', 'systems')} excluded: best value is 0"


def _counted(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def profile_step_curve(
    x: np.ndarray, fraction: np.ndarray, *, x_min: float, x_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points of a right-continuous step curve from 0 to the success rate, ending at ``x_max``.

    ``x`` ascends and ``fraction`` is the share of planned systems solved within it. The last
    point repeats the final fraction at the shared ``x_max``, so the plateau (the method's success
    rate) is drawn up to the right edge. A method that solved nothing is a flat zero line over the
    shared range.
    """
    if not len(x):
        return np.array([x_min, x_max]), np.zeros(2)
    xs = np.concatenate([[x[0]], x, [x_max]])
    ys = np.concatenate([[0.0], fraction, [fraction[-1]]])
    return xs, ys


def _step_profile_figure(
    series: ProfileSeries,
    *,
    style: GroupStyle,
    xlabel: str,
    ylabel: str,
    note: str | None,
    save_path: Path,
    saved_message: str,
) -> Path | None:
    """Draw each method's curve plateauing at its success rate; `None` without a planned system."""
    counts = series.planned.filter(pl.col("n_planned") > 0)
    if counts.height == 0:
        return None
    names = counts["method"].to_list()
    styles = _series_styles(names, style)
    solved = series.frame["x"].to_numpy()
    x_min, x_max = (float(solved.min()), float(solved.max())) if len(solved) else (1.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in counts.iter_rows(named=True):
        name = row["method"]
        curve = series.frame.filter(pl.col("method") == name)
        xs, ys = profile_step_curve(
            curve["x"].to_numpy(), curve["fraction"].to_numpy(), x_min=x_min, x_max=x_max
        )
        ax.step(
            xs,
            ys,
            where="post",
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            label=profile_legend(
                style.legend.label(name), n_solved=row["n_solved"], n_planned=row["n_planned"]
            ),
        )
    if note is not None:
        ax.plot([], [], " ", label=note)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    draw_legend(ax, style.legend, inside_loc="lower right")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message=saved_message,
    )
    return save_path


def plot_performance_profile(
    series: ProfileSeries,
    *,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Dolan-Moré performance profile: fraction of systems within a factor tau of the best method.

    Reference: Dolan & Moré 2002, "Benchmarking optimization software with
    performance profiles", Math. Program. 91. A curve that plateaus below 1
    belongs to a method that failed on some systems; the plateau is its success
    rate over the systems that have a ratio. The legend says how many systems
    were left out because the best value on them is 0.
    """
    return _step_profile_figure(
        series,
        style=style,
        xlabel="Amortized online cost relative to the best method, tau",
        ylabel="Fraction of systems with ratio <= tau",
        note=excluded_zero_best_note(series.n_excluded_zero_best),
        save_path=save_path,
        saved_message="Saved performance profile to",
    )


def plot_ecdf_profile(
    series: ProfileSeries,
    *,
    style: GroupStyle,
    save_path: Path,
    xlabel: str,
) -> Path | None:
    """Unnormalised ECDF: fraction of systems solved within an absolute budget.

    Reference: Hansen et al. 2016, "COCO: Performance Assessment",
    arXiv:1605.03560. It is not the Moré & Wild (2009) data profile, whose
    budget is divided by n_p + 1. It is independent of the other methods,
    unlike the performance profile, and the plateau is the plain success rate
    over every system the method was run on.
    """
    return _step_profile_figure(
        series,
        style=style,
        xlabel=xlabel,
        ylabel="Fraction of systems solved within budget",
        note=None,
        save_path=save_path,
        saved_message="Saved ECDF profile to",
    )


type TotalColumn = Literal["solve_total_s", "workload_online_s"]
"""The column of a workload-totals table a bar figure draws."""


def plot_workload_bars(
    totals: pl.DataFrame,
    *,
    column: TotalColumn,
    xlabel: str,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Draw one workload total per method; a total is not a per-system amortized share.

    ``totals`` is the workload-totals table. A method that attempted fewer systems than were
    planned is marked incomplete, so its short total is not read as a cheap one.
    """
    if totals.height == 0:
        return None
    rows = list(totals.iter_rows(named=True))
    names = [row["method_id"] for row in rows]
    styles = _series_styles(names, style)
    labels = [style.legend.label(name) for name in names]
    values = [row[column] for row in rows]
    fig, ax = plt.subplots(figsize=_bar_figure_size(labels))
    positions = np.arange(len(names))
    bars = ax.barh(positions, values, color=[styles[name]["color"] for name in names])
    ax.set_yticks(positions, labels, fontsize=LEGEND_FONT_SIZE)
    for bar, row in zip(bars, rows, strict=True):
        suffix = " (incomplete)" if row["n_attempted_rhs"] < row["n_planned_rhs"] else ""
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {row['n_converged']}/{row['n_planned_rhs']}{suffix}",
            va="center",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Method")
    ax.grid(True, axis="x", alpha=0.3)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message="Saved workload bars to",
    )
    return save_path


def _paired_axes_limits(times: Sequence[float]) -> tuple[float, float, float]:
    """Linear non-negative shared axis and its failure-marker position."""
    high = max(times)
    upper = high * _FAILED_AXIS_MARGIN if high > 0 else 1.0
    return 0.0, upper, upper


def plot_paired_time_scatter(
    paired: pl.DataFrame,
    *,
    reference_name: PreconditionerName,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Time of each method against the reference method's time, system by system.

    ``paired`` is a ``paired_systems`` frame of the amortized online time. The reference is the
    method explicitly selected in configuration. A point below the diagonal is a system on
    which the method beat the reference. Where only one of the two failed the
    point sits on the failing axis's edge as a hollow marker; where both
    failed there is nothing to compare and the system is not in the frame. Both axes use
    amortized online time: solve plus the system's planned share of matrix setup.
    """
    all_times = pl.concat([paired["reference_value"], paired["method_value"]]).drop_nulls()
    if all_times.len() == 0:
        return None
    low, high, edge = _paired_axes_limits(all_times.to_list())
    others = paired["method"].unique().sort().to_list()
    styles = _series_styles(others, style)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([low, high], [low, high], color="gray", linewidth=1, linestyle="--")
    for name in others:
        pairs = paired.filter(pl.col("method") == name)
        both = pairs.filter(pl.col("pair_kind") == PairKind.BOTH.value)
        method_only = pairs.filter(pl.col("pair_kind") == PairKind.METHOD_ONLY.value)
        reference_only = pairs.filter(pl.col("pair_kind") == PairKind.REFERENCE_ONLY.value)
        color, marker = styles[name]["color"], styles[name]["marker"]
        ax.scatter(
            both["reference_value"].to_list(),
            both["method_value"].to_list(),
            color=color, marker=marker, s=28,
            label=f"{style.legend.label(name)} ({both.height + method_only.height} solved)",
        )  # fmt: skip
        _hollow(
            ax,
            [edge] * method_only.height,
            method_only["method_value"].to_list(),
            color=color,
            marker=marker,
        )
        _hollow(
            ax,
            reference_only["reference_value"].to_list(),
            [edge] * reference_only.height,
            color=color,
            marker=marker,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(
        f"{style.legend.label(reference_name)} (reference): amortized online time [s]"
        "\nhollow markers: the other axis's run did not converge"
    )
    ax.set_ylabel("Method: amortized online time [s]")
    ax.grid(True, which="both", alpha=0.3)
    _legend_outside(ax, style.legend)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message="Saved paired time scatter to",
    )
    return save_path


def speed_ratio_legend(
    label: str,
    *,
    median: float | None,
    ci: tuple[float, float] | None,
    n_ratio: int,
    n_matrices: int,
) -> str:
    """Legend text of a speed-ratio curve: median, interval and what the interval rests on.

    The interval resamples whole matrices, so the matrix count is stated beside the system count:
    sibling right-hand sides of one matrix are not independent observations.
    """
    text = f"{label}: median {median:.2f}" if median is not None else f"{label}: no paired ratio"
    if ci is not None:
        text += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    systems = _counted(n_ratio, "system", "systems")
    matrices = _counted(n_matrices, "matrix", "matrices")
    return f"{text}, n={systems} on {matrices} (reference / method)"


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
    save_path: Path,
) -> Path | None:
    """ECDF of per-system reference/method speed ratios.

    ``ratios`` holds the ``speed_ratios`` rows and ``summary`` the ``speed_ratio_summary`` rows
    of the same pairing, one per method. Only systems both methods solved enter. The median speed
    ratio carries a percentile-bootstrap interval (Efron & Tibshirani 1993) that resamples
    whole matrices, so sibling right-hand sides do not count as independent
    observations. Each legend entry shows the median, its interval and the number of paired
    systems and matrices. The vertical line at 1 separates systems where the method was slower
    (left) from faster (right).
    """
    if summary.height == 0:
        return None
    names = summary["method"].to_list()
    styles = _series_styles(names, style)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary.iter_rows(named=True):
        name = row["method"]
        xs, ys = _speed_ratio_curve(ratios.filter(pl.col("method") == name)["ratio"].to_list())
        ci = None if row["ci_low"] is None else (row["ci_low"], row["ci_high"])
        ax.step(
            xs,
            ys,
            where="post",
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            label=speed_ratio_legend(
                style.legend.label(name),
                median=row["median_ratio"],
                ci=ci,
                n_ratio=row["n_ratio"],
                n_matrices=row["n_matrices"],
            ),
        )
    ax.axvline(1.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(
        f"{style.legend.label(reference_name)} reference / method speed ratio "
        "(slower < 1, faster > 1)"
    )
    ax.set_ylabel("Fraction of paired systems")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=LEGEND_FONT_SIZE)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message="Saved speed ratio CDF to",
    )
    return save_path
