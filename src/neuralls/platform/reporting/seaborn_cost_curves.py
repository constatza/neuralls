"""Cost-curve figures: cumulative cost against the reference, and residual reduction per iteration.

Seaborn draws the marks that have no stored counterpart beyond the rows themselves (the curve
lines, the per-system points); every summary mark (a break-even marker, a mean, a median, an
interval whisker) is read from the frame stored as the figure's table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.domain.benchmarking.cost_curves import BreakEvenState, CostSide
from neuralls.domain.solver.reporting.cohorts import single_cohort
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_sizing import POINT_JITTER
from neuralls.platform.reporting.figure_text import draw_legend
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.platform.reporting.series_style import method_palette
from neuralls.shared.types import PreconditionerName

_REFERENCE_DASHES = (4.0, 2.0)
"""Dash pattern of a reference curve; the compared method's curve in the same colour is solid."""

_BREAK_EVEN_MARKER = "*"
_BREAK_EVEN_MARKER_SIZE = 140


class InconsistentBreakEvenError(ValueError):
    """A cost-curve row states a break-even count that contradicts its break-even state."""


def break_even_legend(label: str, state: BreakEvenState, break_even_rhs: float | None) -> str:
    """Legend text of a compared method's curve: where, or whether, it breaks even.

    A method that never breaks even, or shares no system with the reference, keeps its entry so
    its absence from the crossings is stated rather than left to be inferred.
    """
    match state, break_even_rhs:
        case BreakEvenState.BREAKS_EVEN, float(n_star):
            return f"{label}: breaks even at n* = {n_star:.1f} RHS"
        case BreakEvenState.NEVER, None:
            return f"{label}: never breaks even (not faster per RHS)"
        case BreakEvenState.NO_COMMON_SYSTEMS, None:
            return f"{label}: no systems in common with the reference"
        case _:
            raise InconsistentBreakEvenError(
                f"break-even state {state} contradicts break-even count {break_even_rhs}"
            )


def reference_curve_legend(reference_label: str, method_label: str) -> str:
    """Legend text of the reference's curve paired with one compared method.

    The reference's means are taken over the systems it shares with that method, so each pair has
    its own reference curve.
    """
    return f"{reference_label} (reference, on the systems shared with {method_label})"


def plot_break_even_curves(
    curves: pl.DataFrame,
    *,
    reference_name: PreconditionerName,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    r"""Cumulative cost $C(n)$ of each compared method and of the reference, for $n$ RHS of one matrix.

    ``curves`` is a ``break_even_curves`` frame. Each compared method is a solid line and its
    reference a dashed line of the same colour; a star marks the break-even count $n^*$ where the
    method breaks even. ``None`` when no pair shares a system, so no curve has a cost.
    """
    drawn = curves.filter(pl.col("cost_s").is_not_null())
    if drawn.height == 0:
        return None
    marks = curves.unique("method", keep="first", maintain_order=True)
    names = marks["method"].to_list()
    series = method_palette(names, style)
    palette = {name: series.color(name) for name in names}
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=plot_frame(drawn),
        x="n_rhs",
        y="cost_s",
        hue="method",
        style="side",
        units="method",
        estimator=None,
        errorbar=None,
        palette=palette,
        dashes={CostSide.METHOD.value: "", CostSide.REFERENCE.value: _REFERENCE_DASHES},
        legend=False,
        ax=ax,
    )
    reference_label = style.legend.label(reference_name)
    for row in marks.iter_rows(named=True):
        name, color = row["method"], palette[row["method"]]
        label = style.legend.label(name)
        state = BreakEvenState(row["break_even"])
        text = break_even_legend(label, state, row["break_even_rhs"])
        # Seaborn's own legend would name the raw method ids; empty lines carry the legend text.
        match state:
            case BreakEvenState.NO_COMMON_SYSTEMS:
                ax.plot([], [], " ", label=text)
                continue
            case BreakEvenState.BREAKS_EVEN:
                ax.scatter(
                    [row["break_even_rhs"]],
                    [row["break_even_cost_s"]],
                    color=color,
                    marker=_BREAK_EVEN_MARKER,
                    s=_BREAK_EVEN_MARKER_SIZE,
                    zorder=3,
                )
            case BreakEvenState.NEVER:
                pass
            case unreachable:
                assert_never(unreachable)
        ax.plot([], [], color=color, label=text)
        ax.plot(
            [],
            [],
            color=color,
            dashes=_REFERENCE_DASHES,
            label=reference_curve_legend(reference_label, label),
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Right-hand sides solved on one matrix, n")
    ax.set_ylabel("Cumulative cost: mean setup + n x mean solve [s]")
    ax.grid(True, alpha=0.3)
    draw_legend(ax, style.legend, inside_loc="lower right")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved break-even curves to",
    )
    return target


_POINT_ALPHA = 0.45
_SUMMARY_COLOR = "black"
_MEDIAN_MARKER = "|"
_MEDIAN_MARKER_SIZE = 260
_FACTOR_WIDTH_IN = 7.0
"""Width in inches of the figure, method labels included."""


def plot_convergence_factors(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    target: FigureTarget,
) -> FigureTarget | None:
    """Per-method residual reduction per iteration of the one cohort ``summary`` holds.

    ``summary`` is the rows of one cohort of the figure's cohort summary table; the mean over
    systems with its 95% matrix-cluster interval and the median tick of each method come from it.
    Seaborn strips the per-system points, the ``observations`` of the same cohort the summary was
    taken over, which no table stores again. ``None`` without a row.
    """
    if summary.height == 0:
        return None
    single_cohort(summary)  # one axes draws one cohort; several would overlay their rows
    names = sorted(set(summary["method"].to_list()))
    series = method_palette(names, style)
    palette = {name: series.color(name) for name in names}
    fig, ax = plt.subplots(figsize=(_FACTOR_WIDTH_IN, max(3.0, 1.1 * len(names) + 1.5)))
    points = select(observations, metric, Population(summary["population"][0]))
    _draw_factor_panel(ax, summary, points, names, palette, metric, style)
    ax.set_ylabel("Method")
    ax.plot(
        [], [], "o", color=_SUMMARY_COLOR, label="mean over systems, 95% matrix-cluster interval"
    )
    ax.plot([], [], _MEDIAN_MARKER, color=_SUMMARY_COLOR, markersize=12, label="median")
    draw_legend(ax, style.legend, inside_loc="lower right")
    finish_group_figure(
        fig,
        ax,
        title=style.title,
        subtitle=style.subtitle,
        target=target,
        show=False,
        saved_message="Saved convergence factors to",
    )
    return target


def _draw_factor_panel(
    ax: Axes,
    rows: pl.DataFrame,
    points: pl.DataFrame,
    names: Sequence[str],
    palette: Mapping[str, object],
    metric: MetricId,
    style: GroupStyle,
) -> None:
    """One cohort's points and summary marks, a method per row in the order of ``names``."""
    column = METRICS[metric].column
    by_method = {row["method"]: row for row in rows.iter_rows(named=True)}
    if points.height > 0:
        sns.stripplot(
            data=plot_frame(points.select("method", column)),
            x=column,
            y="method",
            order=list(names),
            hue="method",
            hue_order=list(names),
            palette=dict(palette),
            alpha=_POINT_ALPHA,
            jitter=POINT_JITTER,
            legend=False,
            ax=ax,
        )
    for position, name in enumerate(names):
        # A method that planned no system of this recipe has no row to draw.
        if (row := by_method.get(name)) is not None:
            _draw_summary_marks(ax, position, row)
    ax.set_yticks(range(len(names)), [style.legend.label(name) for name in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(left=0.0)
    ax.set_xlabel(f"{METRICS[metric].label} (smaller converges faster)")
    ax.grid(True, axis="x", alpha=0.3)


def _draw_summary_marks(ax: Axes, position: int, row: Mapping[str, object]) -> None:
    """The mean with its interval whisker and the median tick of one summary row, if it has any."""
    mean_and_interval = (
        row["mean_over_systems"],
        row["mean_over_systems_ci_low"],
        row["mean_over_systems_ci_high"],
    )
    match mean_and_interval:
        case float(mean), float(low), float(high):
            ax.errorbar(
                [mean],
                [position],
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=_SUMMARY_COLOR,
                capsize=4,
                zorder=3,
            )
        case float(mean), None, None:
            # Below two matrices the bootstrap gives no interval, so the mean stands alone.
            ax.scatter([mean], [position], color=_SUMMARY_COLOR, zorder=3)
        case None, None, None:
            return
        case other:
            raise InconsistentSummaryRowError(
                f"summary row has an inconsistent mean and interval: {other!r}"
            )
    match row["q50"]:
        case float(median):
            ax.scatter(
                [median],
                [position],
                marker=_MEDIAN_MARKER,
                s=_MEDIAN_MARKER_SIZE,
                color=_SUMMARY_COLOR,
                zorder=3,
            )
        case _:
            return


class InconsistentSummaryRowError(ValueError):
    """A summary row has an interval without a mean, or only one end of an interval."""
