"""Cost-curve figures: cumulative cost against the reference, and residual reduction per iteration.

Seaborn draws the marks that have no stored counterpart beyond the rows themselves (the curve
lines, the per-system points); every summary mark (a break-even marker, a mean, a median, an
interval whisker) is read from the frame stored as the figure's table.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from neuralls.domain.benchmarking.cost_curves import BreakEvenState, CostSide
from neuralls.platform.reporting.figure_text import draw_legend
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.platform.reporting.plots import finish_group_figure, group_series_styles
from neuralls.platform.reporting.seaborn_figures import plot_frame
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
