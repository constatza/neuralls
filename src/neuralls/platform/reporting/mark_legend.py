"""One legend that explains the summary marks a figure actually drew.

A drawer records which marks it drew and hands that set here, so a mark that is absent from a
figure (an interval needs ``MIN_INTERVAL_MATRICES`` matrices) is never explained. Each entry is a
proxy artist whose glyph looks like the drawn mark, with plain-language text and no counts.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import Enum, auto
from typing import assert_never

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.typing import ColorType

from neuralls.platform.reporting.figure_text import LEGEND_ANCHOR, LEGEND_FONT_SIZE

_INK: ColorType = "black"
_BAR_FACE: ColorType = "0.7"
_GLYPH_SIZE = 9.0

SYSTEM_POINT_STYLE: dict[str, object] = {
    "color": "white",
    "edgecolor": _INK,
    "linewidth": 0.5,
    "alpha": 0.85,
}
"""Seaborn arguments of the per-system points: white with a dark edge, visible on any bar colour."""

_I_BEAM = MarkerStyle(
    Path(
        [(-0.5, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, -1.0), (-0.5, -1.0), (0.5, -1.0)],
        [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO],
    ),
    fillstyle="none",
)
"""A vertical line with a cap at each end: the glyph of an interval whisker."""


class Mark(Enum):
    """A summary mark a figure can draw; the order is the order of the legend entries."""

    MEAN_BAR = auto()
    RATIO_BAR = auto()
    CONFIDENCE_INTERVAL = auto()
    IQR = auto()
    MEDIAN = auto()
    MEDIAN_POINT = auto()
    VIOLIN = auto()
    QUANTILE_WHISKER = auto()
    MEAN_POINT = auto()
    SYSTEM_POINTS = auto()
    MATRIX_MEAN_POINTS = auto()

    @property
    def text(self) -> str:
        """The plain-language legend text of the mark."""
        match self:
            case Mark.MEAN_BAR | Mark.MEAN_POINT:
                return "mean"
            case Mark.RATIO_BAR:
                return "ratio of sums"
            case Mark.CONFIDENCE_INTERVAL:
                return "95% matrix-cluster interval"
            case Mark.IQR:
                return "IQR (q25-q75)"
            case Mark.MEDIAN | Mark.MEDIAN_POINT:
                return "median"
            case Mark.VIOLIN:
                return "density of systems"
            case Mark.QUANTILE_WHISKER:
                return "whiskers within q05-q95"
            case Mark.SYSTEM_POINTS:
                return "systems"
            case Mark.MATRIX_MEAN_POINTS:
                return "matrix means"
            case unreachable:
                assert_never(unreachable)


def _proxy(mark: Mark) -> Line2D | Patch:
    """A legend artist that looks like ``mark`` drawn on a figure."""
    label = mark.text
    match mark:
        case Mark.MEAN_BAR | Mark.RATIO_BAR:
            return Patch(facecolor=_BAR_FACE, edgecolor=_INK, linewidth=0.5, label=label)
        case Mark.IQR:
            return Rectangle((0, 0), 1, 1, fill=False, edgecolor=_INK, linewidth=1.0, label=label)
        case Mark.MEDIAN:
            return Line2D(
                [],
                [],
                marker="_",
                markersize=_GLYPH_SIZE,
                markeredgewidth=1.5,
                linestyle="none",
                color=_INK,
                label=label,
            )
        case Mark.MEDIAN_POINT:
            return Line2D(
                [],
                [],
                marker="D",
                markersize=5.0,
                linestyle="none",
                color=_INK,
                label=label,
            )
        case Mark.VIOLIN:
            return Patch(
                facecolor=_BAR_FACE, edgecolor=_INK, linewidth=0.5, alpha=0.45, label=label
            )
        case Mark.QUANTILE_WHISKER:
            return Line2D(
                [],
                [],
                marker="|",
                markersize=_GLYPH_SIZE,
                markeredgewidth=1.0,
                linestyle="none",
                color=_INK,
                label=label,
            )
        case Mark.CONFIDENCE_INTERVAL:
            return Line2D(
                [],
                [],
                marker=_I_BEAM,
                markersize=_GLYPH_SIZE,
                markeredgewidth=1.0,
                linestyle="none",
                color=_INK,
                label=label,
            )
        case Mark.MEAN_POINT:
            return Line2D(
                [],
                [],
                marker="o",
                markersize=5.0,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=_INK,
                markeredgewidth=1.2,
                label=label,
            )
        case Mark.SYSTEM_POINTS:
            return Line2D(
                [],
                [],
                marker="o",
                markersize=4.0,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=_INK,
                markeredgewidth=0.5,
                label=label,
            )
        case Mark.MATRIX_MEAN_POINTS:
            return Line2D(
                [],
                [],
                marker="o",
                markersize=4.0,
                linestyle="none",
                color=_INK,
                alpha=0.7,
                label=label,
            )
        case unreachable:
            assert_never(unreachable)


def mark_handles(marks: Collection[Mark]) -> list[Line2D | Patch]:
    """Legend artists for ``marks``, one each, in the order of ``Mark``."""
    return [_proxy(mark) for mark in Mark if mark in marks]


def draw_mark_legend(ax: Axes, marks: Collection[Mark]) -> None:
    """Draw the legend of ``marks`` outside ``ax`` on the right, where no mark or label sits."""
    handles = mark_handles(marks)
    if not handles:
        return
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=LEGEND_ANCHOR, fontsize=LEGEND_FONT_SIZE
    )
