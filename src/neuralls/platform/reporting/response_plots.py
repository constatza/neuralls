"""Response figure: each method family's metric against the one fact it sweeps.

One panel per swept fact (a side and a fact key), sharing the metric axis; one line per family
in a panel, its marker at the median and its bar from the 25th to the 75th percentile, all read
from the response table stored beside the figure. A method that never converged keeps its row
in the table but draws nothing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator

from neuralls.domain.benchmarking.response import LOG_X_SPAN
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE
from neuralls.platform.reporting.figure_writer import finish_group_figure
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.series_style import color_pool
from neuralls.shared.scientific_description.models import FieldKey
from neuralls.shared.scientific_description.render import render_symbol

_PANEL_SIZE_IN = (4.0, 4.0)
"""Width and height in inches of one swept fact's panel."""

_MARKER = "o"
_MARKER_SIZE = 5.0
_CAP_SIZE = 3.0
_GRID_ALPHA = 0.3
_PANEL_KEYS = ["side", "fact"]


def plot_response(
    response: pl.DataFrame, *, metric: MetricId, style: GroupStyle, target: FigureTarget
) -> FigureTarget | None:
    """Draw one cohort's response table; ``None`` when it has no defined median."""
    drawn = response.filter(pl.col("q50").is_not_null()).sort("family", "x", maintain_order=True)
    if drawn.height == 0:
        return None
    families = drawn["family"].unique(maintain_order=True).to_list()
    colors = dict(zip(families, color_pool(len(families)), strict=True))
    panels = drawn.select(_PANEL_KEYS).unique(maintain_order=True).rows()
    width, height = _PANEL_SIZE_IN
    fig, axes = plt.subplots(
        1, len(panels), sharey=True, squeeze=False, figsize=(width * len(panels), height)
    )
    for ax, (side, fact) in zip(axes[0], panels, strict=True):
        rows = drawn.filter((pl.col("side") == side) & (pl.col("fact") == fact))
        _draw_panel(ax, rows, colors)
        ax.set_xlabel(render_symbol(FieldKey(fact), style.notation))
    first = axes[0][0]
    first.set_ylabel(METRICS[metric].label.capitalize())
    finish_group_figure(
        fig,
        first,
        title=style.title,
        subtitle=style.subtitle,
        show_titles=style.show_titles,
        target=target,
        show=False,
        saved_message="Saved response figure to",
    )
    return target


def _draw_panel(ax: Axes, rows: pl.DataFrame, colors: dict[str, object]) -> None:
    """One line per family of ``rows``, sorted by x, ticked at the swept values themselves."""
    for (family,), line in rows.group_by("family", maintain_order=True):
        q25, q50, q75 = line["q25"], line["q50"], line["q75"]
        ax.errorbar(
            line["x"].to_list(),
            q50.to_list(),
            yerr=[(q50 - q25).to_list(), (q75 - q50).to_list()],
            color=colors[str(family)],
            marker=_MARKER,
            markersize=_MARKER_SIZE,
            capsize=_CAP_SIZE,
            label=str(family),
        )
    xs = sorted(set(rows["x"].to_list()))
    if xs[0] > 0 and xs[-1] / xs[0] >= LOG_X_SPAN:
        ax.set_xscale("log")
    ax.set_xticks(xs, labels=[f"{x:g}" for x in xs])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, alpha=_GRID_ALPHA)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
