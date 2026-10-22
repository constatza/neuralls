"""What a report sink needs, besides the numbers, to draw one group."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle


@dataclass(frozen=True, slots=True)
class SolverTimingContext:
    """Solver, normalization, and repeat settings needed to interpret a figure."""

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: str
    m_max: int
    reference_precision_margin: float
    timing_repeats: int
    timing_warmup: int
    timing_statistic: str
    normalization: str


@dataclass(frozen=True, slots=True)
class GroupPresentation:
    """Everything a sink needs, besides the numbers, to draw one report group.

    Attributes:
        subject: What the group is, for titles (e.g. ``"ds, gaussian"`` or
            ``"All datasets"``).
        subtitle: Concise report scope shown below the metric title.
        plot_styles: Style of each preconditioner of the group, by name.
        solver_contexts: Distinct solver/timing configurations represented by
            systems in the group.
    """

    subject: str
    subtitle: str
    plot_styles: Mapping[str, PreconditionerPlotStyle]
    solver_contexts: tuple[SolverTimingContext, ...] = ()
