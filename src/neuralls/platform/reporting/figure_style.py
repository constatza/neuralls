"""The one matplotlib style report figures are drawn in, and the label helpers that belong to it."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from neuralls.platform.reporting.figure_sizing import wrap_axis_label
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE

TITLE_FONT_SIZE = 13
"""Point size of a figure's title."""

SUBTITLE_FONT_SIZE = 9
"""Point size of the scope line under a figure's title and of per-panel titles."""

AXIS_LABEL_FONT_SIZE = 9
"""Point size of axis labels."""

TICK_FONT_SIZE = LEGEND_FONT_SIZE
"""Point size of tick labels, equal to the legend entries so method labels read alike everywhere."""

SAVE_DPI = 200
"""Resolution of every saved PNG."""

_SERIF_FAMILY = "serif"
_AXES_LINE_WIDTH = 0.6
_DATA_LINE_WIDTH = 1.0


def _apply_paper_sizes() -> None:
    """Set the fixed sizes, line widths and spines on top of the seaborn theme."""
    plt.rc("font", size=TICK_FONT_SIZE)
    plt.rc("axes", titlesize=SUBTITLE_FONT_SIZE, labelsize=AXIS_LABEL_FONT_SIZE)
    plt.rc("axes", linewidth=_AXES_LINE_WIDTH)
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("xtick", labelsize=TICK_FONT_SIZE)
    plt.rc("ytick", labelsize=TICK_FONT_SIZE)
    plt.rc("legend", fontsize=LEGEND_FONT_SIZE)
    plt.rc("figure", titlesize=TITLE_FONT_SIZE)
    plt.rc("lines", linewidth=_DATA_LINE_WIDTH)


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
