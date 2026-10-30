"""The conditions under which one planned system was solved."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.group_presentation import SolverTimingContext


@dataclass(frozen=True, slots=True)
class SystemProvenance:
    """A planned system with the solver settings and reference state it was timed under.

    Persisted with the raw events because the reference is only known at run time and a
    case config edited later would describe different numerics than the ones timed.
    """

    tag: SystemTag
    solver: SolverTimingContext
    reference: ReferenceSelection
