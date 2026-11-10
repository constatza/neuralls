"""How one method's systems are drawn in a distribution figure, chosen from how many there are.

A density or a box estimated from a handful of systems is mostly noise, so the sample size picks
the mark: the systems alone, a box over them, or a violin over a box. Pure and free of
matplotlib, so the thresholds are testable without a figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RowLimits:
    """The sample sizes from which a distribution row gains a box and a violin.

    Validated once, where the configuration is parsed (``FigureStyleConfig``), so that
    ``1 <= min_box_samples <= min_violin_samples`` holds here.
    """

    min_box_samples: int
    min_violin_samples: int


@dataclass(frozen=True, slots=True)
class PointsOnly:
    """Too few systems for any summary shape: the points and a median tick."""


@dataclass(frozen=True, slots=True)
class BoxAndSwarm:
    """Enough systems for a box, too few for a density: a box over a beeswarm of the systems."""


@dataclass(frozen=True, slots=True)
class Violin:
    """Enough systems for a density: a violin with a box inside it."""


type RowMode = PointsOnly | BoxAndSwarm | Violin


def row_mode(n_samples: int, limits: RowLimits) -> RowMode:
    """The mode of a row of ``n_samples`` systems under ``limits``."""
    if n_samples >= limits.min_violin_samples:
        return Violin()
    if n_samples >= limits.min_box_samples:
        return BoxAndSwarm()
    return PointsOnly()
