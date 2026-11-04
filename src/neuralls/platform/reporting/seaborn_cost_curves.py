"""Cost-curve figures: cumulative cost against the reference, and residual reduction per iteration.

Seaborn draws the marks that have no stored counterpart beyond the rows themselves (the curve
lines, the per-system points); every summary mark (a break-even marker, a mean, a median, an
interval whisker) is read from the frame stored as the figure's table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.domain.benchmarking.cost_curves import BreakEvenState, CostSide
from neuralls.domain.presentation.recipes import render_recipe
from neuralls.domain.solver.reporting.cohorts import COHORT_COLUMN, MIXTURE_COHORT
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.platform.reporting.figure_text import TITLE_PROFILE, draw_legend
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.platform.reporting.plots import finish_group_figure, group_series_styles
from neuralls.platform.reporting.seaborn_figures import plot_frame
from neuralls.shared.rhs_recipe import parse_recipe_key
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
    save_path: Path,
) -> Path | None:
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
    styles = group_series_styles(names, style.families, style.color_keys, style.marker_keys)
    palette = {name: styles[name]["color"] for name in names}
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
        save_path=save_path,
        show=False,
        saved_message="Saved break-even curves to",
    )
    return save_path


_POINT_ALPHA = 0.45
_POINT_JITTER = 0.2
_SUMMARY_COLOR = "black"
_MEDIAN_MARKER = "|"
_MEDIAN_MARKER_SIZE = 260
_FACTOR_PANEL_WIDTH_IN = 7.0
"""Width in inches of one recipe panel, method labels included."""
_HEADER_RATIO = 0.02
"""Height of the header row that carries the scope above all panels, relative to a panel."""


def factor_axis_label(label: str, *, n_defined: int, n_planned: int) -> str:
    """Method-axis text of a convergence-factor row: the method and how many systems have a factor.

    Only converged systems with a stored history have one, so the count keeps the others visible.
    """
    return f"{label}\n({n_defined}/{n_planned} systems with a factor)"


def plot_convergence_factors(
    summary: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    metric: MetricId,
    style: GroupStyle,
    save_path: Path,
) -> Path | None:
    """Per-method residual reduction per iteration, one panel per recipe cohort of ``summary``.

    ``summary`` is the figure's cohort summary table; the mean over systems with its 95%
    matrix-cluster interval and the median tick of each method come from it. Seaborn strips the
    per-system points, the observation rows of the panel's recipe the summary was taken over, which
    no table stores again. The pooled rows of a mixture stay in the table only, since pooling
    hides how the recipes differ. ``None`` without a recipe cohort.
    """
    drawn = summary.filter(pl.col(COHORT_COLUMN) != MIXTURE_COHORT)
    cohorts = drawn[COHORT_COLUMN].unique(maintain_order=True).to_list()
    if not cohorts:
        return None
    names = sorted(set(drawn["method"].to_list()))
    styles = group_series_styles(names, style.families, style.color_keys, style.marker_keys)
    palette = {name: styles[name]["color"] for name in names}
    fig = plt.figure(
        figsize=(_FACTOR_PANEL_WIDTH_IN * len(cohorts), max(3.0, 1.1 * len(names) + 1.5))
    )
    grid = fig.add_gridspec(2, len(cohorts), height_ratios=(_HEADER_RATIO, 1.0))
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    axes = [fig.add_subplot(grid[1, column]) for column in range(len(cohorts))]
    for column, (ax, cohort) in enumerate(zip(axes, cohorts, strict=True)):
        rows = drawn.filter(pl.col(COHORT_COLUMN) == cohort)
        points = select(observations, metric, Population(rows["population"][0])).filter(
            pl.col("recipe") == cohort
        )
        _draw_factor_panel(ax, rows, points, names, palette, metric, style)
        ax.set_ylabel("Method" if column == 0 else "")
        if len(cohorts) > 1:
            recipe = parse_recipe_key(cohort)
            ax.set_title(render_recipe(recipe, TITLE_PROFILE, style.notation), fontsize=9)
    axes[-1].plot(
        [], [], "o", color=_SUMMARY_COLOR, label="mean over systems, 95% matrix-cluster interval"
    )
    axes[-1].plot([], [], _MEDIAN_MARKER, color=_SUMMARY_COLOR, markersize=12, label="median")
    draw_legend(axes[-1], style.legend, inside_loc="lower right")
    finish_group_figure(
        fig,
        header,
        title=style.title,
        subtitle=style.subtitle,
        save_path=save_path,
        show=False,
        saved_message="Saved convergence factors to",
    )
    return save_path


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
            jitter=_POINT_JITTER,
            legend=False,
            ax=ax,
        )
    labels = []
    for position, name in enumerate(names):
        row = by_method.get(name)
        if row is None:
            # The method planned no system of this recipe, so it has no row to draw or count.
            labels.append(style.legend.label(name))
            continue
        _draw_summary_marks(ax, position, row)
        labels.append(
            factor_axis_label(
                style.legend.label(name), n_defined=row["n_defined"], n_planned=row["n_planned"]
            )
        )
    ax.set_yticks(range(len(names)), labels)
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
