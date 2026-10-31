"""Decoding a serialized event frame returns the events that were serialized."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    UnknownPeakMemory,
)
from neuralls.domain.solver.reporting.event_frames import (
    SOLVE_ATTEMPT_SCHEMA,
    build_setup_frame,
    build_solve_frame,
    decode_setup_frame,
    decode_solve_frame,
)
from neuralls.shared.device import Agreed, Diverged, RepeatAgreement, SingleRun, Unchecked
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId
from tests.measurement_support import NO_REFERENCE, NO_RESIDUAL, SINGLE_RUN
from tests.recipe_support import GAUSSIAN


@pytest.fixture
def setup_events() -> tuple[MatrixSetupEvent, ...]:
    dataset, method = DatasetId("ds"), PreconditionerName("jacobi")
    return (
        MatrixSetupEvent(
            SetupEventId("e0"),
            dataset,
            MatrixId(0),
            method,
            0.5,
            (0.4, 0.6),
            MeasuredPeakMemory(10),
            SetupCompleted(),
        ),
        MatrixSetupEvent(
            SetupEventId("e1"),
            dataset,
            MatrixId(1),
            method,
            0.0,
            (),
            UnknownPeakMemory(),
            SetupFailed("singular"),
        ),
    )


@pytest.fixture
def solve_attempts() -> tuple[SolveAttempt, ...]:
    common = (DatasetId("ds"), GAUSSIAN, MatrixId(0), 0, PreconditionerName("jacobi"))
    return (
        SolveAttempt(
            SetupEventId("e0"),
            SystemId("s0"),
            *common,
            0.1,
            (0.1,),
            MeasuredPeakMemory(5),
            SolveConverged(7, 1e-9),
            NO_RESIDUAL,
            NO_REFERENCE,
            SINGLE_RUN,
        ),
        SolveAttempt(
            SetupEventId("e0"),
            SystemId("s1"),
            *common,
            0.2,
            (0.2,),
            UnknownPeakMemory(),
            SolveFailed(20, "max iterations"),
            NO_RESIDUAL,
            NO_REFERENCE,
            SINGLE_RUN,
        ),
    )


def test_setup_events_survive_a_frame_round_trip(
    setup_events: tuple[MatrixSetupEvent, ...],
) -> None:
    assert decode_setup_frame(build_setup_frame(setup_events)) == setup_events


def test_solve_attempts_survive_a_frame_round_trip(
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    assert decode_solve_frame(build_solve_frame(solve_attempts)) == solve_attempts


@pytest.mark.parametrize(
    "repeats", [SingleRun(), Unchecked(), Agreed(), Diverged(first=12, other=13)]
)
def test_every_repeat_agreement_survives_a_frame_round_trip(
    solve_attempts: tuple[SolveAttempt, ...], repeats: RepeatAgreement
) -> None:
    attempts = tuple(replace(attempt, repeats=repeats) for attempt in solve_attempts)

    assert decode_solve_frame(build_solve_frame(attempts)) == attempts


@pytest.mark.parametrize(
    "columns",
    [
        {"repeat_status": "agreed", "repeat_first_iterations": 3},
        {"repeat_status": "diverged"},
        {"repeat_status": "diverged", "repeat_first_iterations": 5, "repeat_other_iterations": 5},
        {"repeat_status": "sometimes"},
    ],
)
def test_inconsistent_repeat_columns_are_rejected(
    solve_attempts: tuple[SolveAttempt, ...], columns: dict[str, object]
) -> None:
    frame = build_solve_frame(solve_attempts)
    for name, value in columns.items():
        frame = frame.with_columns(pl.lit(value, dtype=SOLVE_ATTEMPT_SCHEMA[name]).alias(name))

    with pytest.raises(ValueError):
        decode_solve_frame(frame)
