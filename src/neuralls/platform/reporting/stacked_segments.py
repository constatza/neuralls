"""The key of a stacked bar's segments, shared by the workload and cost-breakdown bars."""

from __future__ import annotations

from matplotlib.patches import Patch

SEGMENT_KEY_COLOR = "gray"
"""Colour of a segment key whose bars carry each method's colour."""


def segment_key(
    label: str, alpha: float, hatch: str | None, color: str = SEGMENT_KEY_COLOR
) -> Patch:
    """The legend patch of one segment: ``color`` (neutral unless the bars share one), opacity, hatch."""
    return Patch(
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        hatch=hatch,
        label=label,
    )
