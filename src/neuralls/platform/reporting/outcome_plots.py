"""Group-level plots that keep failed runs visible instead of averaging them away.

A failed run (iteration cap or breakdown) has no time-to-solution, so none of
these plots gives it a value: profiles leave it unsolved (Dolan & Moré 2002,
Math. Program. 91; Hansen et al. 2016, arXiv:1605.03560) and the scatters pin
it to an axis edge as a hollow marker. Every plot function takes the path to
save to and returns it, or `None` when there was nothing to draw. The curves
and points themselves come from `domain.benchmarking.profiles` and
`domain.benchmarking.paired`; this module only draws them.

Time plots use amortized online seconds: solve plus the planned share of the
matrix setup event.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from neuralls.domain.benchmarking.costs import Cost, amortized_online_seconds, iterations
from neuralls.domain.benchmarking.paired import PairedPoints, paired_points
from neuralls.domain.benchmarking.profiles import Profile, ecdf_profiles, performance_profiles
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    ReferenceSelection,
)
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.outcomes import Failed, Outcomes, Solved, SystemOutcome
from neuralls.domain.solver.models.statistics import PairedComparison
from neuralls.platform.reporting.plots import finish_group_figure, group_series_styles
from neuralls.shared.types import PreconditionerName, SystemId

_FAILED_AXIS_MARGIN = 1.6
"""Factor above the slowest solved run at which failures are drawn; a position on the
axis marking "did not converge", not a time."""


@dataclass(frozen=True, slots=True)
class GroupStyle:
    """How a group's preconditioners are titled and drawn, shared by every plot here."""

    title: str
    subtitle: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    families: Mapping[str, Hashable] = field(default_factory=dict)
    color_keys: Mapping[str, Hashable] = field(default_factory=dict)
    marker_keys: Mapping[str, Hashable] = field(default_factory=dict)


def plot_aggregate_bars(
    outcome: GroupAggregated,
    *,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Plot explicit convergence counts for every method in a report group."""
    if not outcome.convergence:
        return None
    names = list(outcome.convergence)
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [outcome.convergence[name].n_converged for name in names]
    bars = ax.barh(names, values)
    ax.set_xlim(0, max(1, outcome.n_planned_rhs))
    ax.set_xlabel("Converged RHS (of planned RHS)")
    ax.set_ylabel("Method")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, name in zip(bars, names):
        summary = outcome.convergence[name]
        bar_label = f"{summary.n_converged}/{summary.n_attempted_rhs}/{summary.n_planned_rhs}"
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


def _legend_outside(ax: Axes) -> None:
    """Place the legend right of the axes so it never covers the points."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)


def _hollow(ax: Axes, xs: Sequence[float], ys: Sequence[float], *, color: Any, marker: Any) -> None:
    """Draw unfilled markers: the run behind each did not converge, so its value is an axis edge."""
    ax.scatter(xs, ys, facecolors="none", edgecolors=color, marker=marker, s=28)


def _shared_x_range(profiles: Mapping[str, Profile]) -> tuple[float, float]:
    """Smallest and largest value over every profile, `(1, 1)` when no system was solved."""
    values = [v for profile in profiles.values() for v in profile.solved_at]
    return (min(values), max(values)) if values else (1.0, 1.0)


def _step_curve(profile: Profile, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Points of a right-continuous step curve from 0 to the success rate, ending at `x_max`.

    The last point repeats the final fraction at the shared `x_max`, so the
    plateau (the method's success rate) is drawn up to the right edge. A
    profile that solved nothing is a flat zero line over the shared range.
    """
    values = np.asarray(profile.solved_at, dtype=np.float64)
    if not len(values):
        return np.array([x_min, x_max]), np.zeros(2)
    fractions = np.arange(1, len(values) + 1) / profile.n_systems
    xs = np.concatenate([[values[0]], values, [x_max]])
    ys = np.concatenate([[0.0], fractions, [fractions[-1]]])
    return xs, ys


def _step_profile_figure(
    profiles: Mapping[str, Profile],
    *,
    style: GroupStyle,
    xlabel: str,
    ylabel: str,
    log_x: bool,
    save_path: Path,
    saved_message: str,
) -> Path | None:
    """Draw each profile as a step curve plateauing at its success rate; `None` without profiles."""
    if not profiles:
        return None
    names = list(profiles)
    styles = _series_styles(names, style)
    x_min, x_max = _shared_x_range(profiles)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        profile = profiles[name]
        xs, ys = _step_curve(profile, x_min, x_max)
        ax.step(
            xs,
            ys,
            where="post",
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            label=(
                f"{style.labels.get(name, name)} "
                f"({len(profile.solved_at)}/{profile.n_systems} solved)"
            ),
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
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
    outcomes: Outcomes,
    *,
    style: GroupStyle,
    save_path: Path,
    cost: Cost = amortized_online_seconds,
) -> Path | None:
    """Dolan-Moré performance profile: fraction of systems within a factor tau of the best method.

    Reference: Dolan & Moré 2002, "Benchmarking optimization software with
    performance profiles", Math. Program. 91. A curve that plateaus below 1
    belongs to a method that failed on some systems; the plateau is its success
    rate over the systems some method solved. The default cost is amortized
    online time (solve plus allocated matrix setup).
    """
    return _step_profile_figure(
        performance_profiles(outcomes, cost),
        style=style,
        xlabel="Amortized online cost relative to the best method, tau",
        ylabel="Fraction of systems with ratio <= tau",
        log_x=True,
        save_path=save_path,
        saved_message="Saved performance profile to",
    )


def plot_ecdf_profile(
    outcomes: Outcomes,
    *,
    style: GroupStyle,
    save_path: Path,
    cost: Cost = iterations,
    xlabel: str = "Iterations",
) -> Path | None:
    """Unnormalised ECDF: fraction of systems solved within an absolute budget.

    Reference: Hansen et al. 2016, "COCO: Performance Assessment",
    arXiv:1605.03560. It is not the Moré & Wild (2009) data profile, whose
    budget is divided by n_p + 1. It is independent of the other methods,
    unlike the performance profile, and the plateau is the plain success rate
    over every system the method was run on. When a time cost is selected, the
    budget is one of the explicit solve-only or amortized-online scenarios.
    """
    return _step_profile_figure(
        ecdf_profiles(outcomes, cost),
        style=style,
        xlabel=xlabel,
        ylabel="Fraction of systems solved within budget",
        log_x=True,
        save_path=save_path,
        saved_message="Saved ECDF profile to",
    )


def _solved_times(per_system: Mapping[SystemId, SystemOutcome]) -> dict[SystemId, float]:
    return {
        sid: outcome.amortized_online_s
        for sid, outcome in per_system.items()
        if isinstance(outcome, Solved)
    }


def plot_time_scatter(
    outcomes: Outcomes,
    *,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Per-system amortized online time, failures as hollow markers on the top edge.

    Unlike a median bar, every system stays a point, so a method that is fast
    only because it fails is visible as a row of hollow markers. The y value is
    solve time plus the system's planned share of matrix setup.
    Returns `None` when there is nothing to place: no solved system has a recorded time
    and none failed (a log axis cannot be drawn without data).
    """
    if not outcomes:
        return None
    names = list(outcomes)
    styles = _series_styles(names, style)
    system_ids = list(dict.fromkeys(sid for per_system in outcomes.values() for sid in per_system))
    position = {sid: i for i, sid in enumerate(system_ids)}
    all_times = [t for per_system in outcomes.values() for t in _solved_times(per_system).values()]
    any_failed = any(
        isinstance(outcome, Failed)
        for per_system in outcomes.values()
        for outcome in per_system.values()
    )
    if not all_times and not any_failed:
        return None
    failed_y = _FAILED_AXIS_MARGIN * max(all_times) if all_times else 1.0

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        per_system = outcomes[name]
        solved = _solved_times(per_system)
        failed = [sid for sid, outcome in per_system.items() if isinstance(outcome, Failed)]
        color, marker = styles[name]["color"], styles[name]["marker"]
        ax.scatter(
            [position[sid] for sid in solved],
            list(solved.values()),
            color=color,
            marker=marker,
            s=28,
            label=f"{style.labels.get(name, name)} ({len(solved)}/{len(per_system)} solved)",
        )
        _hollow(
            ax,
            [position[sid] for sid in failed],
            [failed_y] * len(failed),
            color=color,
            marker=marker,
        )
    ax.set_yscale("log")
    ax.set_xlabel("System (plan order)\nhollow markers: did not converge")
    ax.set_ylabel("Amortized online time: solve + allocated matrix setup [s]")
    ax.grid(True, which="both", alpha=0.3)
    _legend_outside(ax)
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message="Saved per-system time scatter to",
    )
    return save_path


def _paired_axes_limits(times: Sequence[float]) -> tuple[float, float, float]:
    """Axis (low, high, failure position) for a paired scatter, failures beyond the slowest run."""
    low, high = min(times), max(times)
    return low / _FAILED_AXIS_MARGIN, high * _FAILED_AXIS_MARGIN**2, high * _FAILED_AXIS_MARGIN


def _all_paired_times(points: Mapping[PreconditionerName, PairedPoints]) -> list[float]:
    """Every time that appears on a paired scatter, for sizing its axes."""
    return [
        t
        for p in points.values()
        for t in (
            *(x.amortized_online_s for pair in p.both for x in pair),
            *(x.amortized_online_s for x in p.method_only),
            *(x.amortized_online_s for x in p.reference_only),
        )
        if t > 0
    ]


def _positive_paired_points(points: PairedPoints) -> PairedPoints:
    """Keep only solved coordinates that are valid inputs to logarithmic axes."""
    return PairedPoints(
        both=tuple(
            (reference, method)
            for reference, method in points.both
            if reference.amortized_online_s > 0 and method.amortized_online_s > 0
        ),
        method_only=tuple(point for point in points.method_only if point.amortized_online_s > 0),
        reference_only=tuple(
            point for point in points.reference_only if point.amortized_online_s > 0
        ),
    )


def plot_paired_time_scatter(
    outcomes: Outcomes,
    *,
    selection: ReferenceSelection | None = None,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Time of each method against the reference method's time, system by system.

    The reference is the method explicitly selected in configuration. A point below the diagonal is a system on
    which the method beat the reference. Where only one of the two failed the
    point sits on the failing axis's edge as a hollow marker; where both
    failed, or a solved run has no comparable time, there is nothing to
    compare and the system is not drawn. Both axes use amortized online time:
    solve plus the system's planned share of matrix setup.
    """
    if not isinstance(selection, ConfiguredReference):
        return None
    reference = selection.name
    if reference not in outcomes:
        return None
    others = [name for name in outcomes if name != reference]
    raw_points = {name: paired_points(outcomes[reference], outcomes[name]) for name in others}
    points = {name: _positive_paired_points(point) for name, point in raw_points.items()}
    all_times = _all_paired_times(points)
    if not all_times:
        return None
    low, high, edge = _paired_axes_limits(all_times)
    styles = _series_styles(others, style)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([low, high], [low, high], color="gray", linewidth=1, linestyle="--")
    for name in others:
        p = points[name]
        raw_p = raw_points[name]
        color, marker = styles[name]["color"], styles[name]["marker"]
        ax.scatter(
            [ref.amortized_online_s for ref, _ in p.both],
            [own.amortized_online_s for _, own in p.both],
            color=color, marker=marker, s=28,
            label=f"{style.labels.get(name, name)} ({len(raw_p.both) + len(raw_p.method_only)} solved)",
        )  # fmt: skip
        _hollow(
            ax,
            [edge] * len(p.method_only),
            [point.amortized_online_s for point in p.method_only],
            color=color,
            marker=marker,
        )
        _hollow(
            ax,
            [point.amortized_online_s for point in p.reference_only],
            [edge] * len(p.reference_only),
            color=color,
            marker=marker,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(
        f"{style.labels.get(reference, reference)} (reference): amortized online time [s]"
        "\nhollow markers: the other axis's run did not converge"
    )
    ax.set_ylabel("Method: amortized online time [s]")
    ax.grid(True, which="both", alpha=0.3)
    _legend_outside(ax)
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


def _speed_ratio_legend(label: str, comparison: PairedComparison) -> str:
    """Legend text: median speed ratio with its interval and pair count."""
    median = comparison.median_speed_ratio
    text = f"{label}: median {median:.2f}" if median is not None else f"{label}: no paired ratio"
    if comparison.median_speed_ratio_ci is not None:
        ci = comparison.median_speed_ratio_ci
        text += f" [{ci.low:.2f}, {ci.high:.2f}]"
    if comparison.wilcoxon_p is not None:
        text += f", Wilcoxon p={comparison.wilcoxon_p:.3g}"
    return f"{text}, n={comparison.n_both_solved}"


def _speed_ratio_curve(comparison: PairedComparison) -> tuple[np.ndarray, np.ndarray]:
    """Sorted reference/method ratios and their cumulative fractions."""
    ratios = np.sort(np.asarray(comparison.speed_ratios, dtype=np.float64))
    fractions = np.arange(1, len(ratios) + 1) / len(ratios) if len(ratios) else np.asarray([])
    return ratios, fractions


def plot_speed_ratio_cdf(
    comparisons: Mapping[PreconditionerName, PairedComparison],
    *,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """ECDF of per-system reference/method speed ratios.

    Only systems both methods solved enter. The median speed ratio carries a
    percentile-bootstrap interval (Efron & Tibshirani 1993) and the Wilcoxon
    signed-rank p-value (Wilcoxon 1945) tests the log ratios against zero. The
    vertical line at 1 separates systems where the method was slower (left)
    from faster (right).
    """
    if not comparisons:
        return None
    names = list(comparisons)
    styles = _series_styles(names, style)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        xs, ys = _speed_ratio_curve(comparisons[name])
        ax.step(
            xs,
            ys,
            where="post",
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"],
            label=_speed_ratio_legend(style.labels.get(name, name), comparisons[name]),
        )
    ax.axvline(1.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Reference/method speed ratio")
    ax.set_ylabel("Fraction of paired systems")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
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
