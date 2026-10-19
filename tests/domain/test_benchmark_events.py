"""Tests for typed raw setup and solve benchmark events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId


@pytest.fixture
def dataset_id() -> DatasetId:
    """A stable comparison dataset identity."""
    return DatasetId("ds")


@pytest.fixture
def matrix_id() -> MatrixId:
    """A physical matrix identity within the comparison dataset."""
    return MatrixId(7)


@pytest.fixture
def system_id() -> SystemId:
    """A stable solved-system identity."""
    return SystemId("ds-7-0")


@pytest.fixture
def method() -> PreconditionerName:
    """A preconditioner identity."""
    return PreconditionerName("amg")


@pytest.fixture
def setup_event_id() -> SetupEventId:
    """A stable setup-event identity."""
    return SetupEventId("amg:ds:7")


@pytest.fixture
def measured_memory() -> MeasuredPeakMemory:
    """A known peak-memory measurement."""
    return MeasuredPeakMemory(bytes=4096)


@pytest.fixture
def setup_event(
    dataset_id: DatasetId,
    matrix_id: MatrixId,
    method: PreconditionerName,
    measured_memory: MeasuredPeakMemory,
    setup_event_id: SetupEventId,
) -> MatrixSetupEvent:
    """A completed matrix setup event retaining every raw observation."""
    return MatrixSetupEvent(
        event_id=setup_event_id,
        dataset=dataset_id,
        matrix_id=matrix_id,
        method=method,
        setup_s=1.25,
        samples_s=(1.0, 1.25, 1.5),
        peak_memory=measured_memory,
        outcome=SetupCompleted(),
    )


@pytest.fixture
def solve_attempt(
    dataset_id: DatasetId,
    matrix_id: MatrixId,
    method: PreconditionerName,
    measured_memory: MeasuredPeakMemory,
    setup_event_id: SetupEventId,
    system_id: SystemId,
) -> SolveAttempt:
    """A converged solve attempt retaining every raw observation."""
    return SolveAttempt(
        setup_event_id=setup_event_id,
        system_id=system_id,
        dataset=dataset_id,
        recipe="sinusoidal ω=6.28",
        matrix_id=matrix_id,
        rhs_ordinal=0,
        method=method,
        solve_s=0.75,
        samples_s=(0.5, 0.75, 1.0),
        peak_memory=measured_memory,
        outcome=SolveConverged(iterations=4, final_relative_residual=1e-9),
    )


def test_setup_event_retains_the_complete_raw_payload(setup_event: MatrixSetupEvent) -> None:
    """Dropping a setup identity, timing, or measurement makes raw aggregation impossible."""
    assert setup_event.event_id == SetupEventId("amg:ds:7")
    assert setup_event.dataset == DatasetId("ds")
    assert setup_event.matrix_id == MatrixId(7)
    assert setup_event.method == PreconditionerName("amg")
    assert setup_event.setup_s == 1.25
    assert setup_event.samples_s == (1.0, 1.25, 1.5)
    assert setup_event.peak_memory == MeasuredPeakMemory(bytes=4096)
    assert setup_event.outcome == SetupCompleted()


def test_solve_attempt_retains_the_complete_raw_payload(solve_attempt: SolveAttempt) -> None:
    """Dropping a solve link, timing, or outcome makes workload reporting ambiguous."""
    assert solve_attempt.setup_event_id == SetupEventId("amg:ds:7")
    assert solve_attempt.system_id == SystemId("ds-7-0")
    assert solve_attempt.dataset == DatasetId("ds")
    assert solve_attempt.recipe == "sinusoidal ω=6.28"
    assert solve_attempt.matrix_id == MatrixId(7)
    assert solve_attempt.rhs_ordinal == 0
    assert solve_attempt.method == PreconditionerName("amg")
    assert solve_attempt.solve_s == 0.75
    assert solve_attempt.samples_s == (0.5, 0.75, 1.0)
    assert solve_attempt.peak_memory == MeasuredPeakMemory(bytes=4096)
    assert solve_attempt.outcome == SolveConverged(iterations=4, final_relative_residual=1e-9)


@pytest.mark.parametrize("event_fixture", ["setup_event", "solve_attempt"])
def test_events_are_frozen_and_slotted(request: pytest.FixtureRequest, event_fixture: str) -> None:
    """Mutable or dictionary-backed event records could silently corrupt benchmark facts."""
    event = request.getfixturevalue(event_fixture)

    assert not hasattr(event, "__dict__")
    with pytest.raises(FrozenInstanceError):
        event.method = PreconditionerName("jacobi")


@pytest.mark.parametrize(
    ("constructor", "arguments"),
    [
        (
            MatrixSetupEvent,
            {
                "dataset": DatasetId("ds"),
                "matrix_id": MatrixId(7),
                "method": PreconditionerName("amg"),
                "setup_s": 1.0,
                "samples_s": (1.0,),
                "peak_memory": UnknownPeakMemory(),
                "outcome": SetupCompleted(),
            },
        ),
        (
            MatrixSetupEvent,
            {
                "event_id": SetupEventId("amg:ds:7"),
                "dataset": DatasetId("ds"),
                "matrix_id": MatrixId(7),
                "method": PreconditionerName("amg"),
                "samples_s": (1.0,),
                "peak_memory": UnknownPeakMemory(),
                "outcome": SetupCompleted(),
            },
        ),
        (
            MatrixSetupEvent,
            {
                "event_id": SetupEventId("amg:ds:7"),
                "dataset": DatasetId("ds"),
                "matrix_id": MatrixId(7),
                "method": PreconditionerName("amg"),
                "setup_s": 1.0,
                "peak_memory": UnknownPeakMemory(),
                "outcome": SetupCompleted(),
            },
        ),
        (
            MatrixSetupEvent,
            {
                "event_id": SetupEventId("amg:ds:7"),
                "dataset": DatasetId("ds"),
                "matrix_id": MatrixId(7),
                "method": PreconditionerName("amg"),
                "setup_s": 1.0,
                "samples_s": (1.0,),
                "outcome": SetupCompleted(),
            },
        ),
        (
            SolveAttempt,
            {
                "system_id": SystemId("ds-7-0"),
                "dataset": DatasetId("ds"),
                "recipe": "gauss",
                "matrix_id": MatrixId(7),
                "rhs_ordinal": 0,
                "method": PreconditionerName("amg"),
                "solve_s": 1.0,
                "samples_s": (1.0,),
                "peak_memory": UnknownPeakMemory(),
                "outcome": SolveConverged(iterations=1, final_relative_residual=0.0),
            },
        ),
        (
            SolveAttempt,
            {
                "setup_event_id": SetupEventId("amg:ds:7"),
                "system_id": SystemId("ds-7-0"),
                "dataset": DatasetId("ds"),
                "recipe": "gauss",
                "matrix_id": MatrixId(7),
                "rhs_ordinal": 0,
                "method": PreconditionerName("amg"),
                "samples_s": (1.0,),
                "peak_memory": UnknownPeakMemory(),
                "outcome": SolveConverged(iterations=1, final_relative_residual=0.0),
            },
        ),
        (
            SolveAttempt,
            {
                "setup_event_id": SetupEventId("amg:ds:7"),
                "system_id": SystemId("ds-7-0"),
                "dataset": DatasetId("ds"),
                "recipe": "gauss",
                "matrix_id": MatrixId(7),
                "rhs_ordinal": 0,
                "method": PreconditionerName("amg"),
                "solve_s": 1.0,
                "peak_memory": UnknownPeakMemory(),
                "outcome": SolveConverged(iterations=1, final_relative_residual=0.0),
            },
        ),
        (
            SolveAttempt,
            {
                "setup_event_id": SetupEventId("amg:ds:7"),
                "system_id": SystemId("ds-7-0"),
                "dataset": DatasetId("ds"),
                "recipe": "gauss",
                "matrix_id": MatrixId(7),
                "rhs_ordinal": 0,
                "method": PreconditionerName("amg"),
                "solve_s": 1.0,
                "samples_s": (1.0,),
                "outcome": SolveConverged(iterations=1, final_relative_residual=0.0),
            },
        ),
    ],
)
def test_events_require_identity_timing_samples_and_memory(
    constructor: type[MatrixSetupEvent | SolveAttempt], arguments: dict[str, object]
) -> None:
    """Defaults for raw identities or measurements would fabricate benchmark data."""
    with pytest.raises(TypeError):
        constructor(**arguments)


def test_solve_outcomes_are_disjoint() -> None:
    """A failed solve has a reason, while a converged solve has a final residual."""
    with pytest.raises(TypeError):
        SolveConverged(iterations=4, final_relative_residual=1e-9, reason="iteration limit")
    with pytest.raises(TypeError):
        SolveFailed(iterations=100, reason="iteration limit", final_relative_residual=1e-9)


def test_setup_outcomes_are_disjoint() -> None:
    """A completed setup has no failure reason and a failed setup must retain one."""
    with pytest.raises(TypeError):
        SetupCompleted(reason="factorization failed")
    with pytest.raises(TypeError):
        missing_reason: dict[str, object] = {}
        SetupFailed(**missing_reason)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeasuredPeakMemory(bytes=-1),
        lambda: SolveConverged(iterations=-1, final_relative_residual=0.0),
        lambda: SolveConverged(iterations=1, final_relative_residual=-1.0),
        lambda: SolveFailed(iterations=-1, reason="iteration limit"),
        lambda: SystemTag(
            dataset=DatasetId("ds"),
            recipe="gauss",
            matrix_id=MatrixId(7),
            rhs_ordinal=-1,
            system_id=SystemId("ds-7-0"),
        ),
    ],
)
def test_event_components_reject_negative_quantities(factory: Callable[[], object]) -> None:
    """Negative measurements and ordinals are invalid raw benchmark facts."""
    with pytest.raises(ValueError):
        factory()
