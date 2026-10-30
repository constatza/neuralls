"""Event reductions replacing legacy projected-cost aggregation."""

from __future__ import annotations

from dataclasses import replace

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.labels import AllMatrixSetsLabel
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.models.statistics import IterationStats
from neuralls.domain.solver.reporting.aggregation import convergence_by_level, outcomes_by_system
from neuralls.domain.solver.reporting.report_levels import AllMatrixSetsLevel
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)
from tests.recipe_support import GAUSSIAN


def _tag(ordinal: int) -> SystemTag:
    return SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(7), ordinal, SystemId(f"ds-7-{ordinal}"))


def _setup(method: PreconditionerName) -> MatrixSetupEvent:
    return MatrixSetupEvent(
        SetupEventId(f"{method}:ds:7"),
        DatasetId("ds"),
        MatrixId(7),
        method,
        6.0,
        (6.0,),
        UnknownPeakMemory(),
        SetupCompleted(),
    )


def _attempt(tag: SystemTag, method: PreconditionerName, solve_s: float) -> SolveAttempt:
    return SolveAttempt(
        SetupEventId(f"{method}:ds:7"),
        tag.system_id,
        tag.dataset,
        tag.recipe,
        tag.matrix_id,
        tag.rhs_ordinal,
        method,
        solve_s,
        (solve_s,),
        UnknownPeakMemory(),
        SolveConverged(iterations=tag.rhs_ordinal + 2, final_relative_residual=1e-9),
    )


def test_outcomes_use_system_cost_frame_allocation() -> None:
    method = PreconditionerName("amg")
    planned = tuple(_tag(index) for index in range(3))
    attempts = tuple(_attempt(tag, method, float(index)) for index, tag in enumerate(planned, 1))

    result = outcomes_by_system(planned, (_setup(method),), attempts, AllMatrixSetsLevel())

    assert result[AllMatrixSetsLabel((GAUSSIAN,))][method] == {
        planned[0].system_id: Solved(
            MatrixId(7), 2, solve_s=1.0, amortized_online_s=3.0, dataset="ds"
        ),
        planned[1].system_id: Solved(
            MatrixId(7), 3, solve_s=2.0, amortized_online_s=4.0, dataset="ds"
        ),
        planned[2].system_id: Solved(
            MatrixId(7), 4, solve_s=3.0, amortized_online_s=5.0, dataset="ds"
        ),
    }


def test_failure_time_remains_raw_but_is_censored_from_outcomes() -> None:
    method = PreconditionerName("amg")
    planned = (_tag(0),)
    failed = replace(
        _attempt(planned[0], method, 9.0),
        outcome=SolveFailed(iterations=100, reason="iteration limit"),
    )

    result = outcomes_by_system(planned, (_setup(method),), (failed,), AllMatrixSetsLevel())
    outcome = result[AllMatrixSetsLabel((GAUSSIAN,))][method][planned[0].system_id]

    assert failed.solve_s == 9.0
    assert outcome == Failed(MatrixId(7), 100, "iteration limit")
    assert not hasattr(outcome, "solve_s")


def test_convergence_summary_separates_counts_and_converged_iterations() -> None:
    method = PreconditionerName("amg")
    planned = tuple(_tag(index) for index in range(4))
    attempts = (
        _attempt(planned[0], method, 1.0),
        _attempt(planned[1], method, 2.0),
        replace(
            _attempt(planned[2], method, 3.0),
            outcome=SolveFailed(iterations=100, reason="iteration limit"),
        ),
    )

    result = convergence_by_level(planned, (_setup(method),), attempts, AllMatrixSetsLevel())
    summary = result[AllMatrixSetsLabel((GAUSSIAN,))][method]

    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (4, 3, 2)
    assert summary.iterations == IterationStats(mean=2.5, median=2.5, std=2**-0.5)
    assert not hasattr(summary, "ert_seconds")
