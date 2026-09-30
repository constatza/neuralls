"""Plotting helpers for diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_condition_boxes(
    labels: Iterable[str],
    raw_values: Iterable[float],
    preconditioned_values: Iterable[float],
    *,
    save_path: Path,
) -> Path:
    """Plot boxplots of original vs preconditioned condition numbers.

    Args:
        labels: Iterable of labels for the x-axis.
        raw_values: Iterable of raw condition numbers.
        preconditioned_values: Iterable of preconditioned condition numbers.
        save_path: Path to save the plot to.

    Returns:
        Path to the saved plot.
    """
    labels_list = list(labels)
    raw_list = list(raw_values)
    precond_list = list(preconditioned_values)
    positions = []
    data = []
    colors = ["#4c78a8", "#f58518"]
    for idx, (raw, precond) in enumerate(zip(raw_list, precond_list)):
        start = 2.0 * idx
        positions.extend([start, start + 0.6])
        data.extend([[raw], [precond]])
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels_list) * 1.6), 6.0))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors * len(labels_list)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    tick_positions = [2.0 * idx + 0.3 for idx in range(len(labels_list))]
    ax.set_xticks(tick_positions, labels_list, rotation=20, ha="right")
    ax.set_ylabel("Condition number (2-norm)")
    ax.set_yscale("log")
    ax.set_title("Original vs preconditioned condition numbers")
    ax.legend(
        [bp["boxes"][0], bp["boxes"][1]],
        ["Original", "Preconditioned"],
        loc="best",
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_condition_bars(
    labels: Iterable[str],
    cond_maps: Iterable[dict[str, float]],
    preconditioners: Iterable[str],
    *,
    save_path: Path,
    log_scale: bool = True,
) -> Path:
    """Plot bar chart of condition numbers by preconditioner.

    Args:
        labels: Iterable of labels for the x-axis.
        cond_maps: Iterable of dictionaries mapping preconditioner name to condition number.
        preconditioners: Iterable of preconditioner names.
        save_path: Path to save the plot to.
        log_scale: Whether to use a log scale for the y-axis.

    Returns:
        Path to the saved plot.
    """
    labels_list = list(labels)
    cond_list = list(cond_maps)
    preconds = list(preconditioners)

    categories = [label for label in labels_list for _ in preconds]
    series = preconds * len(labels_list)
    values = [conds.get(name, np.nan) for conds in cond_list for name in preconds]

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels_list) * 1.6), 6.0))
    sns.barplot(x=categories, y=values, hue=series, order=labels_list, hue_order=preconds, ax=ax)
    ax.tick_params(axis="x", rotation=20)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Condition number (2-norm)")
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_title("Condition numbers by preconditioner")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
