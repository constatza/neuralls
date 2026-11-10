"""Sizes and spacings shared by the report figures' drawers."""

from __future__ import annotations

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

_BAR_AREA_WIDTH_IN = 6.0
"""Width in inches kept for the bars themselves, however long the method labels are."""

_LABEL_LINE_HEIGHT_IN = 0.3
"""Height in inches one wrapped label line takes on a bar chart's method axis."""


def bar_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size that leaves the bars their own width and gives every label line its own row."""
    lines = [line for label in labels for line in label.split("\n")]
    widest = max((len(line) for line in lines), default=0)
    return (
        _BAR_AREA_WIDTH_IN + widest * LABEL_CHAR_WIDTH_IN,
        max(5.0, len(lines) * _LABEL_LINE_HEIGHT_IN + 2.0),
    )


def method_rows_figure_size(labels: Sequence[str]) -> tuple[float, float]:
    """Figure size of a plot with one row per method label: room for the longest label line."""
    widest = max(len(line) for label in labels for line in label.split("\n"))
    return (
        widest * LABEL_CHAR_WIDTH_IN + _METHOD_ROWS_WIDTH_IN,
        max(_MIN_METHOD_ROWS_HEIGHT_IN, len(labels) * _METHOD_ROW_HEIGHT_IN + 2.0),
    )
