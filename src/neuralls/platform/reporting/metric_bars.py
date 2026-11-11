"""Bar chart of one metric across labelled experiments, for the training and evaluation batches."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_style import AXIS_LABEL_FONT_SIZE, TICK_FONT_SIZE
from neuralls.platform.reporting.figure_text import FigureTitle, Legend, title_text
from neuralls.platform.reporting.figure_writer import finish_group_figure, use_headless_backend


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    *,
    metric_name: str,
    target: FigureTarget | None = None,
    show: bool = False,
    title: str | None = None,
    horizontal: bool = False,
) -> None:
    """Bar chart comparing a single metric across labelled experiments.

    Each bar is labelled with text the caller rendered from a presentation (see
    ``plot_method_bars``); the labels must be unique because they key the bar colours.

    Args:
        labels: Rendered bar labels. Must be unique.
        values: Metric value for each corresponding label.
        metric_name: Human-readable metric description used as y-axis label
            and figure title (e.g. ``"Mean Absolute Error (eval/mae)"``).
        target: Where to save the figure, one file per figure format. Parent directories are created
            automatically if they do not exist.
        show: Whether to call ``plt.show()`` (disabled by default for
            headless / batch use).
        title: Optional title for the plot. Defaults to "Metric Comparison: {metric_name}".
        horizontal: If True, use horizontal bars (barh) and swap axis labels.
    """
    use_headless_backend()
    label_lines = sum(label.count("\n") + 1 for label in labels)
    if horizontal:
        fig, ax = plt.subplots(figsize=(5, max(3, label_lines * 0.6)))
    else:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    order = [
        label for label, _ in sorted(zip(labels, values), key=lambda pair: pair[1], reverse=True)
    ]
    colormap = matplotlib.colormaps["Set1"]
    colors = colormap(np.linspace(0, 0.9, max(len(labels), 1)))
    palette = dict(zip(labels, colors))
    bar_kwargs = {
        "order": order,
        "hue": labels,
        "palette": palette,
        "legend": False,
        "alpha": 0.85,
        "edgecolor": "black",
        "linewidth": 0.7,
        "ax": ax,
    }

    if horizontal:
        sns.barplot(x=values, y=labels, **bar_kwargs)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.set_ylabel("Experiment", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel(metric_name, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        sns.barplot(x=labels, y=values, **bar_kwargs)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.set_xlabel("Experiment", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(metric_name, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3, axis="y")

    finish_group_figure(
        fig,
        ax,
        title=title or f"Metric Comparison: {metric_name}",
        subtitle=None,
        show_titles=True,
        target=target,
        show=show,
        saved_message="Saved metric comparison plot to",
    )


def plot_method_bars(
    legend: Legend,
    names: Sequence[str],
    values: Sequence[float],
    *,
    metric_name: str,
    title: FigureTitle | None = None,
    horizontal: bool = False,
    target: FigureTarget | None = None,
) -> None:
    """Bar chart of one metric per method, each bar labelled by its legend entry.

    The bar labels are rendered from ``legend`` here; a name without an entry raises.
    """
    plot_metric_comparison(
        [legend.label(name) for name in names],
        list(values),
        metric_name=metric_name,
        title=title_text(title) if title is not None else None,
        horizontal=horizontal,
        target=target,
    )
