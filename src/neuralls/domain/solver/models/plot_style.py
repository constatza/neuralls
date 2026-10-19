"""How a preconditioner is drawn, shared by per-run and per-group plots."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class PreconditionerPlotStyle:
    """How one preconditioner is drawn: its legend label and style-axis keys.

    Carried from the run that built the preconditioner to every later plot of
    it, so a group report is styled exactly like the per-run convergence plots.

    Attributes:
        label: Scientific legend label built from the constructed preconditioner.
        family: Plot-style family; same-family preconditioners share a linestyle.
        color_key: Optional colour-axis key; ``None`` falls back to ``family``.
        marker_key: Optional marker-axis key; ``None`` falls back to ``family``.
    """

    label: str
    family: StrEnum
    color_key: str | None = None
    marker_key: str | None = None


@dataclass(frozen=True, slots=True)
class PlotStyleMaps:
    """Per-name mappings in the shape the plot functions take.

    Keys absent from ``color_keys``/``marker_keys`` fall back to the family.
    """

    labels: Mapping[str, str]
    families: Mapping[str, Hashable]
    color_keys: Mapping[str, Hashable]
    marker_keys: Mapping[str, Hashable]


def plot_style_maps(styles: Mapping[str, PreconditionerPlotStyle]) -> PlotStyleMaps:
    """Split per-preconditioner styles into the mappings plots consume.

    The one derivation shared by per-run and per-group plots, so a
    preconditioner keeps the same colour, marker and linestyle everywhere.
    """
    return PlotStyleMaps(
        labels={name: style.label for name, style in styles.items()},
        families={name: style.family for name, style in styles.items()},
        color_keys={
            name: style.color_key for name, style in styles.items() if style.color_key is not None
        },
        marker_keys={
            name: style.marker_key for name, style in styles.items() if style.marker_key is not None
        },
    )
