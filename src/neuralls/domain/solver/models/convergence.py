"""How often a preconditioner converged over a report group."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.solver.models.statistics import IterationStats


@dataclass(frozen=True, slots=True)
class ConvergenceSummary:
    """Planned, attempted, and converged counts plus successful iterations.

    Attributes:
        n_planned_rhs: Systems planned for the report group.
        n_attempted_rhs: Systems with a recorded solve attempt.
        n_converged: Attempts that reached the tolerance within the
            iteration cap.
        iterations: Iteration statistics over the converged runs, `None` when
            nothing converged.
    """

    n_planned_rhs: int
    n_attempted_rhs: int
    n_converged: int
    iterations: IterationStats | None = None

    def __post_init__(self) -> None:
        """Reject contradictory workload counts."""
        if not 0 <= self.n_converged <= self.n_attempted_rhs <= self.n_planned_rhs:
            raise ValueError("counts must satisfy converged <= attempted <= planned")

    @property
    def success_rate(self) -> float:
        """Fraction of systems that converged."""
        return self.n_converged / self.n_planned_rhs if self.n_planned_rhs else 0.0
