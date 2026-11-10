"""The key of a stacked bar's segments, shared by the workload and cost-breakdown bars."""

from __future__ import annotations

from matplotlib.patches import Patch

SEGMENT_KEY_COLOR = "gray"
"""Colour of the segment key; the bars themselves carry each method's colour."""


def segment_key(label: str, alpha: float, hatch: str | None) -> Patch:
    """The legend patch of one segment: neutral colour, the segment's opacity and hatch."""
    return Patch(
        facecolor=SEGMENT_KEY_COLOR,
        edgecolor=SEGMENT_KEY_COLOR,
        alpha=alpha,
        hatch=hatch,
        label=label,
    )
