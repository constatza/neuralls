"""The one matplotlib style report figures are drawn in, and the label helpers that belong to it."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.platform.reporting.figure_sizing import wrap_axis_label
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE

BASE_FONT_SIZE = LEGEND_FONT_SIZE
"""Point size every other font size and line width is a fixed ratio of (tick, legend, notes)."""

TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
"""Point size of a figure's title."""

SUBTITLE_FONT_SIZE = BASE_FONT_SIZE
"""Point size of the scope line under a figure's title and of per-panel titles."""

AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE + 1
"""Point size of axis labels, one step above the tick labels as in a paper's body text."""

TICK_FONT_SIZE = BASE_FONT_SIZE
"""Point size of tick labels, equal to the legend entries so method labels read alike everywhere."""

SAVE_DPI = 200
"""Resolution of every saved PNG."""

_LINE_UNIT = BASE_FONT_SIZE / 18
"""The thinnest stroke in points (0.5 pt at 9 pt text); every line width is a multiple of it."""

THIN_LINE_WIDTH = _LINE_UNIT
"""Grid lines, cell borders and the edges of filled bars, bodies and points."""

AXES_LINE_WIDTH = 1.6 * _LINE_UNIT
"""Spines and tick marks."""

REFERENCE_LINE_WIDTH = 2 * _LINE_UNIT
"""Reference lines (threshold, diagonal, ratio one), whiskers, outlines and hollow-marker edges."""

DATA_LINE_WIDTH = 3 * _LINE_UNIT
"""Data series lines, the strokes a reader follows."""

EMPHASIS_LINE_WIDTH = 4 * _LINE_UNIT
"""A summary stroke drawn over others, such as a median tick across a box."""

_MAJOR_TICK_LENGTH = 0.4 * BASE_FONT_SIZE
_MINOR_TICK_LENGTH = 0.6 * _MAJOR_TICK_LENGTH
_MINOR_TICK_WIDTH = 0.75 * AXES_LINE_WIDTH

_SERIF_FAMILY = "serif"


def _apply_paper_sizes() -> None:
    """Set the fixed sizes, line widths and spines on top of the seaborn theme."""
    plt.rc("font", size=TICK_FONT_SIZE)
    plt.rc("axes", titlesize=SUBTITLE_FONT_SIZE, labelsize=AXIS_LABEL_FONT_SIZE)
    plt.rc("axes", linewidth=AXES_LINE_WIDTH)
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("xtick", labelsize=TICK_FONT_SIZE)
    plt.rc("ytick", labelsize=TICK_FONT_SIZE)
    plt.rc("xtick.major", width=AXES_LINE_WIDTH, size=_MAJOR_TICK_LENGTH)
    plt.rc("ytick.major", width=AXES_LINE_WIDTH, size=_MAJOR_TICK_LENGTH)
    plt.rc("xtick.minor", width=_MINOR_TICK_WIDTH, size=_MINOR_TICK_LENGTH)
    plt.rc("ytick.minor", width=_MINOR_TICK_WIDTH, size=_MINOR_TICK_LENGTH)
    plt.rc("grid", linewidth=THIN_LINE_WIDTH)
    plt.rc("legend", fontsize=LEGEND_FONT_SIZE)
    plt.rc("figure", titlesize=TITLE_FONT_SIZE)
    plt.rc("lines", linewidth=DATA_LINE_WIDTH, markeredgewidth=REFERENCE_LINE_WIDTH)
    plt.rc("patch", linewidth=THIN_LINE_WIDTH)


@contextmanager
def paper_style() -> Generator[None]:
    """Draw figures inside this block in the publication style, then restore matplotlib's state.

    The style is entered by the code that calls a figure's drawer, never by the drawer, so
    one figure family has one style and a drawer stays callable under any style. The palette
    is left alone because every drawer chooses its own series colours.
    """
    with plt.rc_context():
        sns.set_theme(context="paper", style="ticks", font=_SERIF_FAMILY, palette=None)
        _apply_paper_sizes()
        yield


def rotate_x_labels(ax: Axes, degrees: float) -> None:
    """Wrap and rotate ``ax``'s x tick labels, anchored at their right end under their tick.

    The wrap is ``wrap_axis_label``, the one the figure's size was computed from.
    """
    for label in ax.get_xticklabels():
        label.set_text(wrap_axis_label(label.get_text()))
    ax.tick_params(axis="x", labelrotation=degrees)
    if degrees <= 0:
        return
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
