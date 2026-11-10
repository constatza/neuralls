"""Sizes and spacings shared by the report figures' drawers."""

from __future__ import annotations

import math
from collections.abc import Sequence

LABEL_CHAR_WIDTH_IN = 0.055
"""Width in inches of one legend-size character, used to reserve room for method labels."""

POINT_JITTER = 0.2
"""Jitter of the per-system points drawn over mean bars and convergence factors."""

_METHOD_ROWS_WIDTH_IN = 4.5
"""Width in inches of the bars of a figure with one row per method, method labels excluded."""

_METHOD_ROW_HEIGHT_IN = 0.45
"""Height in inches of one method's row."""

_MIN_METHOD_ROWS_HEIGHT_IN = 4.0

_BAR_COLUMN_WIDTH_IN = 1.1
"""Width in inches of one method's column in a bar figure with methods along the x axis."""

_MIN_BAR_COLUMNS_WIDTH_IN = 5.0
"""Narrowest bar figure, so a figure of one or two methods keeps room for its axis labels."""

_BAR_PLOT_HEIGHT_IN = 4.0
"""Height in inches of a vertical bar figure's plotting area, rotated labels excluded."""

_LABEL_LINE_HEIGHT_IN = 0.2
"""Height in inches one wrapped label line takes along the x axis."""


def bar_columns_figure_size(labels: Sequence[str], rotation_deg: float) -> tuple[float, float]:
    """Figure size of bars with one column per method label, labels rotated by ``rotation_deg``.

    The width follows the number of methods only; annotations sit above the bars and never widen
    the figure. The height adds the vertical extent of the longest label line, which grows with
    the rotation, and of the label's stacked lines.
    """
    lines = [label.split("\n") for label in labels]
    widest = max((len(line) for parts in lines for line in parts), default=0)
    stacked = max((len(parts) for parts in lines), default=1)
    angle = math.radians(rotation_deg)
    label_height = widest * LABEL_CHAR_WIDTH_IN * math.sin(
        angle
    ) + stacked * _LABEL_LINE_HEIGHT_IN * math.cos(angle)
    return (
        max(_MIN_BAR_COLUMNS_WIDTH_IN, len(labels) * _BAR_COLUMN_WIDTH_IN),
        _BAR_PLOT_HEIGHT_IN + label_height,
    )


def method_rows_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size of a plot with one row per method label: room for the longest label line."""
    widest = max(len(line) for label in labels for line in label.split("\n"))
    return (
        widest * LABEL_CHAR_WIDTH_IN + _METHOD_ROWS_WIDTH_IN,
        max(_MIN_METHOD_ROWS_HEIGHT_IN, len(labels) * _METHOD_ROW_HEIGHT_IN + 2.0),
    )
