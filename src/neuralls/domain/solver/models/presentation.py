"""What a report sink needs, besides the numbers, to draw one group."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.plot_style import (
    PreconditionerPlotStyle,
)


@dataclass(frozen=True, slots=True)
class GroupPresentation:
    """Everything a sink needs, besides the numbers, to draw one report group.

    Attributes:
        subject: What the group is, for titles (e.g. ``"ds, gaussian"`` or
            ``"ds, mixture of gaussian, sparse[3 nnz]"``).
        subtitle: Level, keys, recipe and system count, as one plot subtitle.
        plot_styles: Style of each preconditioner of the group, by name.
    """

    subject: str
    subtitle: str
    plot_styles: Mapping[str, PreconditionerPlotStyle]
