"""Failure-aware per-system outcomes derived from benchmark events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from neuralls.shared.types import MatrixId, PreconditionerName, SystemId


@dataclass(frozen=True, slots=True)
class Solved:
    """A run that converged, with both supported per-system time scenarios."""

    matrix_id: MatrixId
    iterations: int
    solve_s: float
    amortized_online_s: float

    def __post_init__(self) -> None:
        """Reject impossible profile observations."""
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if any(
            not isfinite(value) or value < 0 for value in (self.solve_s, self.amortized_online_s)
        ):
            raise ValueError("outcome times must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Failed:
    """A run stopped by the iteration cap or a breakdown: a right-censored observation.

    Its measured attempt time remains in the raw event/frame for workload
    totals. It carries no profile cost, so profiles keep it right-censored.
    """

    matrix_id: MatrixId
    iterations: int
    reason: str

    def __post_init__(self) -> None:
        """Reject an impossible failed-attempt iteration count."""
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")


type SystemOutcome = Solved | Failed


type Outcomes = Mapping[PreconditionerName, Mapping[SystemId, SystemOutcome]]
"""Preconditioner name -> {system id -> outcome}."""
