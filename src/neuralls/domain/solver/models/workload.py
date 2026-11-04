"""Setup-aware online-cost summaries for one method workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from sys import float_info
from typing import Protocol

from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    PeakMemory,
    SolveAttempt,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.shared.digest import Digest, Json
from neuralls.shared.types import PreconditionerName, SystemId

_DERIVED_SECONDS_REL_TOLERANCE = 4 * float_info.epsilon
_DERIVED_SECONDS_ABS_TOLERANCE = 0.0


@dataclass(frozen=True, slots=True)
class CompleteWorkloadSummary:
    """Online costs for a method that attempted every planned RHS system.

    ``failed_solve_s`` is the part of ``solve_total_s`` spent on attempts that did not converge;
    the rest is ``converged_solve_s``. Setup is not split: one matrix setup serves every RHS of
    the matrix, converged or not.
    """

    method: PreconditionerName
    n_planned_rhs: int
    n_attempted_rhs: int
    n_converged: int
    allocated_setup_s: float
    solve_total_s: float
    failed_solve_s: float
    workload_online_s: float
    workload_peak_memory: PeakMemory

    @property
    def converged_solve_s(self) -> float:
        """Solve seconds of the attempts that converged."""
        return _converged_solve_s(self)

    def __post_init__(self) -> None:
        """Keep the complete variant exclusive to fully attempted workloads."""
        _validate_summary_fields(self)
        if self.n_planned_rhs != self.n_attempted_rhs:
            raise ValueError("a complete workload must attempt every planned RHS")


@dataclass(frozen=True, slots=True)
class IncompleteWorkloadSummary:
    """Online costs for a method with explicit unattempted systems; split as the complete one."""

    method: PreconditionerName
    n_planned_rhs: int
    n_attempted_rhs: int
    n_converged: int
    allocated_setup_s: float
    solve_total_s: float
    failed_solve_s: float
    workload_online_s: float
    workload_peak_memory: PeakMemory
    missing_system_ids: tuple[SystemId, ...]

    @property
    def converged_solve_s(self) -> float:
        """Solve seconds of the attempts that converged."""
        return _converged_solve_s(self)

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
    def generation_cost(self) -> StageCost: ...

    @property
    def workload_peak_memory(self) -> PeakMemory: ...

    @property
    def config_digest(self) -> Digest: ...

    @property
    def method_config(self) -> Mapping[str, Json]: ...

    @property
    def diagnostics(self) -> tuple[tuple[SystemId, CGComparisonResult], ...]: ...

    @property
    def plot_styles(self) -> Mapping[str, PreconditionerPlotStyle]: ...

    @property
    def presentation(self) -> ExecutedMethod: ...


def _validate_summary_fields(summary: MethodWorkloadSummary) -> None:
    """Validate count ordering and derived online costs at construction."""
    if not 0 <= summary.n_converged <= summary.n_attempted_rhs <= summary.n_planned_rhs:
        raise ValueError("workload counts must satisfy converged <= attempted <= planned")
    costs = (
        summary.allocated_setup_s,
        summary.solve_total_s,
        summary.failed_solve_s,
        summary.workload_online_s,
    )
    if any(not isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("workload costs must be finite and non-negative")
    if not _derived_seconds_match(
        summary.workload_online_s, summary.allocated_setup_s + summary.solve_total_s
    ):
        raise ValueError("workload online time must equal setup plus solve time")
    if summary.failed_solve_s > summary.solve_total_s and not _derived_seconds_match(
        summary.failed_solve_s, summary.solve_total_s
    ):
        raise ValueError("failed-attempt solve time cannot exceed the solve total")


def _converged_solve_s(summary: MethodWorkloadSummary) -> float:
    """The solve total less the failed attempts' time.

    When every attempt failed the two sums may differ in the last bit, so the difference is
    clamped at zero rather than reported as a negative time.
    """
    return max(summary.solve_total_s - summary.failed_solve_s, 0.0)


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
