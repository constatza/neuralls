"""Setup-aware online-cost summaries for one method workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from sys import float_info
from typing import Protocol

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    PeakMemory,
    SolveAttempt,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.shared.digest import Digest, Json
from neuralls.shared.types import PreconditionerName, SystemId

_DERIVED_SECONDS_REL_TOLERANCE = 4 * float_info.epsilon
_DERIVED_SECONDS_ABS_TOLERANCE = 0.0


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
        if len(set(self.missing_system_ids)) != len(self.missing_system_ids):
            raise ValueError("missing system IDs must be distinct")
        if len(self.missing_system_ids) != self.n_planned_rhs - self.n_attempted_rhs:
            raise ValueError("missing system count must equal planned minus attempted RHS")


type MethodWorkloadSummary = CompleteWorkloadSummary | IncompleteWorkloadSummary
"""The complete or explicitly incomplete summary of one method workload."""


class MethodWorkload(Protocol):
    """Domain-facing record required by the method persistence port."""

    @property
    def method(self) -> PreconditionerName: ...

    @property
    def planned(self) -> tuple[SystemTag, ...]: ...

    @property
    def setup_events(self) -> tuple[MatrixSetupEvent, ...]: ...

    @property
    def solve_attempts(self) -> tuple[SolveAttempt, ...]: ...

    @property
    def generation_cost(self) -> StageCost | None: ...

    @property
    def workload_peak_memory(self) -> PeakMemory: ...

    @property
    def config_digest(self) -> Digest: ...

    @property
    def method_config(self) -> Mapping[str, Json]: ...

    @property
    def diagnostics(self) -> tuple[tuple[SystemId, CGComparisonResult], ...]: ...


def _validate_summary_fields(summary: MethodWorkloadSummary) -> None:
    """Validate count ordering and derived online costs at construction."""
    if not 0 <= summary.n_converged <= summary.n_attempted_rhs <= summary.n_planned_rhs:
        raise ValueError("workload counts must satisfy converged <= attempted <= planned")
    costs = (summary.setup_total_s, summary.solve_total_s, summary.workload_online_s)
    if any(not isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("workload costs must be finite and non-negative")
    if not _derived_seconds_match(
        summary.workload_online_s, summary.setup_total_s + summary.solve_total_s
    ):
        raise ValueError("workload online time must equal setup plus solve time")
    expected_average = (
        summary.workload_online_s / summary.n_attempted_rhs if summary.n_attempted_rhs > 0 else None
    )
    if expected_average is None:
        average_matches = summary.online_s_per_attempted_rhs is None
    else:
        average_matches = summary.online_s_per_attempted_rhs is not None and _derived_seconds_match(
            summary.online_s_per_attempted_rhs, expected_average
        )
    if not average_matches:
        raise ValueError("attempted-RHS average must agree with workload online time")


def _derived_seconds_match(left: float, right: float) -> bool:
    """Allow the bounded rounding from separately reduced non-negative costs.

    ``matrix_online_s`` rounds each setup-plus-solve row before ``fsum`` reduces
    it. Four machine epsilons cover that row-rounding total plus the separately
    rounded setup, solve, and final addition, while excluding material errors.
    """
    return isclose(
        left,
        right,
        rel_tol=_DERIVED_SECONDS_REL_TOLERANCE,
        abs_tol=_DERIVED_SECONDS_ABS_TOLERANCE,
    )
