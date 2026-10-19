"""Setup-aware online-cost summaries for one method workload."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from neuralls.domain.solver.models.benchmark_events import PeakMemory
from neuralls.shared.types import PreconditionerName, SystemId


@dataclass(frozen=True, slots=True)
class CompleteWorkloadSummary:
    """Online costs for a method that attempted every planned RHS system."""

    method: PreconditionerName
    n_planned_rhs: int
    n_attempted_rhs: int
    n_converged: int
    setup_total_s: float
    solve_total_s: float
    workload_online_s: float
    online_s_per_attempted_rhs: float | None
    workload_peak_memory: PeakMemory

    def __post_init__(self) -> None:
        """Keep the complete variant exclusive to fully attempted workloads."""
        _validate_summary_fields(self)
        if self.n_planned_rhs != self.n_attempted_rhs:
            raise ValueError("a complete workload must attempt every planned RHS")


@dataclass(frozen=True, slots=True)
class IncompleteWorkloadSummary:
    """Online costs for a method with explicit unattempted systems."""

    method: PreconditionerName
    n_planned_rhs: int
    n_attempted_rhs: int
    n_converged: int
    setup_total_s: float
    solve_total_s: float
    workload_online_s: float
    online_s_per_attempted_rhs: float | None
    workload_peak_memory: PeakMemory
    missing_system_ids: tuple[SystemId, ...]

    def __post_init__(self) -> None:
        """Keep missing membership consistent with the workload counts."""
        _validate_summary_fields(self)
        if not self.missing_system_ids:
            raise ValueError("an incomplete workload must identify missing systems")
        if len(self.missing_system_ids) != self.n_planned_rhs - self.n_attempted_rhs:
            raise ValueError("missing system count must equal planned minus attempted RHS")


type MethodWorkloadSummary = CompleteWorkloadSummary | IncompleteWorkloadSummary
"""The complete or explicitly incomplete summary of one method workload."""


def _validate_summary_fields(summary: MethodWorkloadSummary) -> None:
    """Validate count ordering and derived online costs at construction."""
    if not 0 <= summary.n_converged <= summary.n_attempted_rhs <= summary.n_planned_rhs:
        raise ValueError("workload counts must satisfy converged <= attempted <= planned")
    costs = (summary.setup_total_s, summary.solve_total_s, summary.workload_online_s)
    if any(not isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("workload costs must be finite and non-negative")
    if summary.workload_online_s != summary.setup_total_s + summary.solve_total_s:
        raise ValueError("workload online time must equal setup plus solve time")
    expected_average = (
        summary.workload_online_s / summary.n_attempted_rhs if summary.n_attempted_rhs > 0 else None
    )
    if summary.online_s_per_attempted_rhs != expected_average:
        raise ValueError("attempted-RHS average must agree with workload online time")
