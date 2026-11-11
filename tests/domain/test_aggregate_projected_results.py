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
from neuralls.domain.solver.models.statistics import IterationStats
from neuralls.domain.solver.reporting.aggregation import convergence_by_level
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.report_levels import AllMatrixSetsLevel
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.shared.device import SingleRun
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)
from tests.measurement_support import NO_REFERENCE, NO_RESIDUAL
from tests.observation_support import observations_of_events
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
        NO_RESIDUAL,
        NO_REFERENCE,
        repeats=SingleRun(),
    )


def test_observations_allocate_the_matrix_setup_over_its_planned_rhs() -> None:
    method = PreconditionerName("amg")
    planned = tuple(_tag(index) for index in range(3))
    attempts = tuple(_attempt(tag, method, float(index)) for index, tag in enumerate(planned, 1))

    obs = observations_of_events(planned, (_setup(method),), attempts)

    # setup 6.0 over 3 planned RHS adds 2.0 to each solve time
    assert obs.select("system_id", "iterations", "solve_s", "online_s", "dataset").rows() == [
        (planned[0].system_id, 2, 1.0, 3.0, "ds"),
        (planned[1].system_id, 3, 2.0, 4.0, "ds"),
        (planned[2].system_id, 4, 3.0, 5.0, "ds"),
    ]


def test_failure_time_remains_raw_but_is_excluded_from_every_metric() -> None:
    method = PreconditionerName("amg")
    planned = (_tag(0),)
    failed = replace(
        _attempt(planned[0], method, 9.0),
        outcome=SolveFailed(iterations=100, reason="iteration limit"),
    )
    obs = observations_of_events(planned, (_setup(method),), (failed,))

    assert failed.solve_s == 9.0
    assert obs["outcome"].to_list() == [Outcome.SOLVER_ERROR.value]
    assert obs["solve_s"].to_list() == [9.0]
    assert select(obs, MetricId.SOLVE_S, Population.OWN).height == 0


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
