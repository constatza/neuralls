"""Decoding a serialized event frame returns the events that were serialized."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    MeasuredResidual,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    UnknownPeakMemory,
    UnmeasuredResidual,
)
from neuralls.domain.solver.reporting.event_frames import (
    SOLVE_ATTEMPT_SCHEMA,
    build_setup_frame,
    build_solve_frame,
    decode_setup_frame,
    decode_solve_frame,
    validate_decoded_solve_frame,
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
            MeasuredResidual(rhs_norm=1.0, true_residual_rel=0.5),
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


@pytest.fixture
def one_system_per_solve_outcome(
    solve_attempts: tuple[SolveAttempt, ...],
) -> tuple[SolveAttempt, ...]:
    """A converged, a zero-RHS converged, a capped, and a raised solve, with realistic fields."""
    template = solve_attempts[0]
    measured = MeasuredResidual(rhs_norm=2.0, true_residual_rel=1e-9)
    return (
        replace(
            template,
            system_id=SystemId("converged"),
            outcome=SolveConverged(7, 1e-9),
            residual=measured,
        ),
        replace(
            template,
            system_id=SystemId("zero-rhs"),
            outcome=SolveConverged(0, 0.0),
            residual=UnmeasuredResidual(),
        ),
        replace(
            template,
            system_id=SystemId("capped"),
            outcome=SolveFailed(500, "max iterations"),
            residual=MeasuredResidual(rhs_norm=2.0, true_residual_rel=0.3),
        ),
        replace(
            template,
            system_id=SystemId("capped-at-zero"),
            outcome=SolveFailed(0, "max iterations"),
            residual=MeasuredResidual(rhs_norm=2.0, true_residual_rel=1.0),
        ),
        replace(
            template,
            system_id=SystemId("raised"),
            outcome=SolveFailed(0, "CG solver failed"),
            residual=UnmeasuredResidual(),
        ),
    )


def test_a_frame_with_each_kind_of_solve_outcome_is_accepted(
    one_system_per_solve_outcome: tuple[SolveAttempt, ...],
) -> None:
    validate_decoded_solve_frame(build_solve_frame(one_system_per_solve_outcome))


def _with_columns(frame: pl.DataFrame, system_id: str, columns: dict[str, object]) -> pl.DataFrame:
    """Overwrite columns of one system's row, keeping the schema dtypes."""
    for name, value in columns.items():
        frame = frame.with_columns(
            pl.when(pl.col("system_id") == system_id)
            .then(pl.lit(value, dtype=SOLVE_ATTEMPT_SCHEMA[name]))
            .otherwise(pl.col(name))
            .alias(name)
        )
    return frame


@pytest.mark.parametrize(
    ("system_id", "columns", "rule"),
    [
        ("raised", {"iterations": 5}, "must have 0 iterations"),
        ("capped", {"rhs_norm": 0.0}, "zero RHS norm"),
    ],
)
def test_cross_column_violations_name_the_system_and_the_rule(
    one_system_per_solve_outcome: tuple[SolveAttempt, ...],
    system_id: str,
    columns: dict[str, object],
    rule: str,
) -> None:
    frame = _with_columns(build_solve_frame(one_system_per_solve_outcome), system_id, columns)

    with pytest.raises(ValueError, match=f"{system_id!r}.*{rule}"):
        validate_decoded_solve_frame(frame)
