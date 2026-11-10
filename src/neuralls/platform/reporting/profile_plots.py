"""Performance profile and ECDF of a report group, in which a failed run stays unsolved.

A failed run (iteration cap or breakdown) has no time-to-solution, so a profile gives it no value
and leaves it unsolved (Dolan & Moré 2002, Math. Program. 91; Hansen et al. 2016,
arXiv:1605.03560). The curves are frames from ``domain.benchmarking.profiles``; this module only
draws them. The performance profile uses amortized online seconds: solve plus the planned share of
the matrix setup event.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from neuralls.domain.benchmarking.profiles import ProfileSeries
from neuralls.platform.reporting.figure_annotations import counted
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_text import draw_legend
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import method_palette


def excluded_zero_best_note(n_excluded: int) -> str | None:
    """Legend note on systems a performance profile left out, or ``None`` when there are none.

    A system whose best value is 0 has no ratio for any method, so it is dropped from every
    curve; the note says how many, so the lower plateau is not read as failures.
    """
    if n_excluded == 0:
        return None
    return f"{counted(n_excluded, 'system', 'systems')} excluded: best value is 0"


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
    target: FigureTarget,
    saved_message: str,
) -> FigureTarget | None:
    """Draw each method's curve plateauing at its success rate; `None` without a planned system."""
    counts = series.planned.filter(pl.col("n_planned") > 0)
    if counts.height == 0:
        return None
    names = counts["method"].to_list()
    palette = method_palette(names, style)
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
            color=palette.color(name),
            linestyle=palette.linestyle(name),
            label=style.legend.label(name),
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
        target=target,
        show=False,
        saved_message=saved_message,
    )
    return target


def plot_performance_profile(
    series: ProfileSeries,
    *,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
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
        target=target,
        saved_message="Saved performance profile to",
    )


def plot_ecdf_profile(
    series: ProfileSeries,
    *,
    style: GroupStyle,
    target: FigureTarget,
    xlabel: str,
) -> FigureTarget | None:
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
        target=target,
        saved_message="Saved ECDF profile to",
    )
