"""Workload totals are named costs, distinct from the per-system amortized cost."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.workload_totals import (
    solve_total_seconds,
    workload_online_seconds,
)
from neuralls.domain.solver.models.benchmark_events import UnknownPeakMemory
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
    MethodWorkloadSummary,
)
from neuralls.shared.types import PreconditionerName, SystemId

SETUP_S = 0.5
SOLVE_S = 2.0


@pytest.fixture
def complete() -> MethodWorkloadSummary:
    return CompleteWorkloadSummary(
        method=PreconditionerName("amg"),
        n_planned_rhs=7,
        n_attempted_rhs=7,
        n_converged=7,
        setup_total_s=SETUP_S,
        solve_total_s=SOLVE_S,
        workload_online_s=SETUP_S + SOLVE_S,
        workload_peak_memory=UnknownPeakMemory(),
    )


@pytest.fixture
def incomplete() -> MethodWorkloadSummary:
    return IncompleteWorkloadSummary(
        method=PreconditionerName("amg"),
        n_planned_rhs=7,
        n_attempted_rhs=6,
        n_converged=5,
        setup_total_s=SETUP_S,
        solve_total_s=SOLVE_S,
        workload_online_s=SETUP_S + SOLVE_S,
        workload_peak_memory=UnknownPeakMemory(),
        missing_system_ids=(SystemId("s7"),),
    )


@pytest.mark.parametrize("summary_fixture", ["complete", "incomplete"])
def test_solve_total_excludes_setup(summary_fixture: str, request: pytest.FixtureRequest) -> None:
    summary: MethodWorkloadSummary = request.getfixturevalue(summary_fixture)

    assert solve_total_seconds(summary) == SOLVE_S


@pytest.mark.parametrize("summary_fixture", ["complete", "incomplete"])
def test_workload_online_total_is_setup_plus_solve(
    summary_fixture: str, request: pytest.FixtureRequest
) -> None:
    summary: MethodWorkloadSummary = request.getfixturevalue(summary_fixture)

    assert workload_online_seconds(summary) == SETUP_S + SOLVE_S
