"""Sizes and spacings shared by the report figures' drawers."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence

from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE

_POINTS_PER_INCH = 72
_LABEL_CHAR_WIDTH_EM = 0.566
"""Mean width of a serif label character in ems, measured at the legend size (0.055 in at 7 pt)."""

LABEL_CHAR_WIDTH_IN = LEGEND_FONT_SIZE * _LABEL_CHAR_WIDTH_EM / _POINTS_PER_INCH
"""Width in inches of one legend-size character, used to reserve room for method labels.

Derived from ``LEGEND_FONT_SIZE`` so a change of the label font size resizes the figures that
reserve room for those labels.
"""

POINT_JITTER = 0.2
"""Jitter of the per-system points drawn over mean bars and convergence factors."""

_METHOD_ROWS_WIDTH_IN = 4.5
"""Width in inches of the bars of a figure with one row per method, method labels excluded."""

_METHOD_ROW_HEIGHT_IN = 0.45
"""Height in inches of one method's row."""

_MIN_METHOD_ROWS_HEIGHT_IN = 4.0

_BAR_COLUMN_WIDTH_IN = 0.85
"""Width in inches of one method's column in a bar figure with methods along the x axis."""

COLUMN_TEXT_WIDTH_IN = 0.66
"""Widest text in inches above a bar, under the column width so neighbours' texts never touch."""

_MIN_BAR_COLUMNS_WIDTH_IN = 5.0
"""Narrowest bar figure, so a figure of one or two methods keeps room for its axis labels."""

_BAR_PLOT_HEIGHT_IN = 4.0
"""Height in inches of a vertical bar figure's plotting area, rotated labels excluded."""

_CELL_WIDTH_IN = 1.1
"""Width in inches of one method's heatmap column."""

_CELL_HEIGHT_IN = 0.4
"""Height in inches of one matrix's heatmap row; the figure grows with the number of matrices."""

_HEATMAP_COLORBAR_WIDTH_IN = 2.5
"""Room in inches right of the cells for the colour bar and its label."""

_HEATMAP_FRAME_HEIGHT_IN = 1.5
"""Room in inches above and below the cells for the axis label and padding, rotated labels excluded."""

_MIN_HEATMAP_HEIGHT_IN = 4.0

_LABEL_LINE_HEIGHT_IN = 0.2
"""Height in inches one wrapped label line takes along the x axis."""


AXIS_LABEL_WRAP = 24
"""Characters per line of a method label with no fact boundary to break at, on a rotated axis."""


def wrap_axis_label(label: str) -> str:
    """Break the plain lines of ``label`` at spaces to ``AXIS_LABEL_WRAP`` characters.

    A line with mathtext already ends at a fact boundary (``wrap_at_boundaries``) and may not be
    cut inside a ``$...$`` span, so only lines without ``$`` are wrapped; line breaks are kept.
    """
    return "\n".join(
        line if "$" in line else textwrap.fill(line, AXIS_LABEL_WRAP) for line in label.split("\n")
    )


def rotated_label_overhang_in(labels: Sequence[str], rotation_deg: float) -> float:
    """Inches the rotated labels reach left of the leftmost column's left edge, 0 if none do.

    Labels are anchored at their right end under their tick (``rotate_x_labels``), so the label of
    column ``j`` extends its widest line times the cosine of the rotation to the left of the tick,
    which stands half a column plus ``j`` columns from the axes' left edge. Unrotated labels are
    centred and reach half their width.
    """
    reach = math.cos(math.radians(rotation_deg)) if rotation_deg > 0 else 0.5
    overhangs = (
        max(len(line) for line in label.split("\n")) * LABEL_CHAR_WIDTH_IN * reach
        - (column + 0.5) * _BAR_COLUMN_WIDTH_IN
        for column, label in enumerate(labels)
    )
    return max(0.0, *overhangs)


def rotated_labels_height_in(labels: Sequence[str], rotation_deg: float) -> float:
    """Inches the wrapped, rotated ``labels`` take below the axis.

    It is the vertical extent of the widest label line, which grows with the rotation, plus the
    label's stacked lines.
    """
    lines = [label.split("\n") for label in labels]
    widest = max((len(line) for parts in lines for line in parts), default=0)
    stacked = max((len(parts) for parts in lines), default=1)
    angle = math.radians(rotation_deg)
    return widest * LABEL_CHAR_WIDTH_IN * math.sin(
        angle
    ) + stacked * _LABEL_LINE_HEIGHT_IN * math.cos(angle)


def bar_columns_figure_size(labels: Sequence[str], rotation_deg: float) -> tuple[float, float]:
    """Figure size of bars with one column per method label, labels rotated by ``rotation_deg``.

    ``labels`` are the method labels; they are measured as ``rotate_x_labels`` draws them, wrapped
    by ``wrap_axis_label``. The width follows the number of methods plus whatever the leftmost
    rotated label overhangs the plotting area, so the labels never squeeze the axes; annotations
    sit above the bars and never widen the figure. The height adds ``rotated_labels_height_in``.
    """
    wrapped = [wrap_axis_label(label) for label in labels]
    return (
        max(_MIN_BAR_COLUMNS_WIDTH_IN, len(labels) * _BAR_COLUMN_WIDTH_IN)
        + rotated_label_overhang_in(wrapped, rotation_deg),
        _BAR_PLOT_HEIGHT_IN + rotated_labels_height_in(wrapped, rotation_deg),
    )


def heatmap_figure_size(
    labels: Sequence[str], rotation_deg: float, n_rows: int
) -> tuple[float, float]:
    """Figure size of a heatmap of ``n_rows`` matrices by one column per method label.

    The width adds the rotated labels' overhang to the cells and the margin of the colour bar;
    the height adds the labels' extent below the cells. Figures carry no title, so no room is kept for one.
    """
    wrapped = [wrap_axis_label(label) for label in labels]
    return (
        len(labels) * _CELL_WIDTH_IN
        + _HEATMAP_COLORBAR_WIDTH_IN
        + rotated_label_overhang_in(wrapped, rotation_deg),
        max(_MIN_HEATMAP_HEIGHT_IN, n_rows * _CELL_HEIGHT_IN + _HEATMAP_FRAME_HEIGHT_IN)
        + rotated_labels_height_in(wrapped, rotation_deg),
    )


def method_rows_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size of a plot with one row per method label: room for the longest label line."""
    widest = max(len(line) for label in labels for line in label.split("\n"))
    return (
        widest * LABEL_CHAR_WIDTH_IN + _METHOD_ROWS_WIDTH_IN,
        max(_MIN_METHOD_ROWS_HEIGHT_IN, len(labels) * _METHOD_ROW_HEIGHT_IN + 2.0),
    )


_PAIRED_AXES_SIDE_IN = 5.5
"""Side in inches of the paired scatter's square plotting area."""

_PAIRED_FRAME_WIDTH_IN = 1.4
"""Width the y label and tick labels take left of the paired scatter's axes."""

_PAIRED_FRAME_HEIGHT_IN = 1.2
"""Height the x label and tick labels take around the paired scatter's axes."""


def paired_scatter_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size of the paired scatter, whose equal log axes are a square.

    The plotting square keeps its side and the figure grows around it by the legend, which sits
    right of the axes, and the text above and below, so the square axes box is not shrunk
    inside a figure that is wider or taller than it needs.
    """
    widest = max((len(line) for label in labels for line in label.split("\n")), default=0)
    return (
        _PAIRED_AXES_SIDE_IN + _PAIRED_FRAME_WIDTH_IN + widest * LABEL_CHAR_WIDTH_IN,
        _PAIRED_AXES_SIDE_IN + _PAIRED_FRAME_HEIGHT_IN,
    )


def with_outside_legend(size: tuple[float, float], labels: Sequence[str]) -> tuple[float, float]:
    """``size`` widened by the legend entries ``labels``, which sit right of the axes.

    Without the extra width a wide legend takes the room of the plotting area.
    """
    widest = max((len(line) for label in labels for line in label.split("\n")), default=0)
    return (size[0] + widest * LABEL_CHAR_WIDTH_IN, size[1])
