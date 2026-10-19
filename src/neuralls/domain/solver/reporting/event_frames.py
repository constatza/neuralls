"""Normalized Polars serialization of canonical benchmark events."""

from __future__ import annotations

from collections.abc import Collection
from typing import assert_never

import polars as pl

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    PeakMemory,
    SetupCompleted,
    SetupFailed,
    SetupOutcome,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SolveOutcome,
    UnknownPeakMemory,
)

SETUP_EVENT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "setup_event_id": pl.String,
    "dataset": pl.String,
    "matrix_id": pl.Int64,
    "method": pl.String,
    "setup_s": pl.Float64,
    "samples_s": pl.List(pl.Float64),
    "peak_memory_bytes": pl.Int64,
    "memory_status": pl.String,
    "setup_status": pl.String,
    "failure_reason": pl.String,
}

SOLVE_ATTEMPT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "setup_event_id": pl.String,
    "system_id": pl.String,
    "dataset": pl.String,
    "recipe": pl.String,
    "matrix_id": pl.Int64,
    "rhs_ordinal": pl.Int64,
    "method": pl.String,
    "solve_s": pl.Float64,
    "samples_s": pl.List(pl.Float64),
    "peak_memory_bytes": pl.Int64,
    "memory_status": pl.String,
    "solve_status": pl.String,
    "iterations": pl.Int64,
    "final_relative_residual": pl.Float64,
    "failure_reason": pl.String,
}


def build_setup_frame(events: Collection[MatrixSetupEvent]) -> pl.DataFrame:
    """Serialize setup events without weakening their domain representation."""
    rows: list[dict[str, object]] = []
    for event in events:
        peak_memory_bytes, memory_status = _memory_fields(event.peak_memory)
        setup_status, failure_reason = _setup_outcome_fields(event.outcome)
        rows.append(
            {
                "setup_event_id": event.event_id,
                "dataset": event.dataset,
                "matrix_id": event.matrix_id,
                "method": event.method,
                "setup_s": event.setup_s,
                "samples_s": event.samples_s,
                "peak_memory_bytes": peak_memory_bytes,
                "memory_status": memory_status,
                "setup_status": setup_status,
                "failure_reason": failure_reason,
            }
        )
    return pl.DataFrame(rows, schema=SETUP_EVENT_SCHEMA)


def build_solve_frame(attempts: Collection[SolveAttempt]) -> pl.DataFrame:
    """Serialize solve attempts without weakening their domain representation."""
    rows: list[dict[str, object]] = []
    for attempt in attempts:
        peak_memory_bytes, memory_status = _memory_fields(attempt.peak_memory)
        solve_status, iterations, final_residual, failure_reason = _solve_outcome_fields(
            attempt.outcome
        )
        rows.append(
            {
                "setup_event_id": attempt.setup_event_id,
                "system_id": attempt.system_id,
                "dataset": attempt.dataset,
                "recipe": attempt.recipe,
                "matrix_id": attempt.matrix_id,
                "rhs_ordinal": attempt.rhs_ordinal,
                "method": attempt.method,
                "solve_s": attempt.solve_s,
                "samples_s": attempt.samples_s,
                "peak_memory_bytes": peak_memory_bytes,
                "memory_status": memory_status,
                "solve_status": solve_status,
                "iterations": iterations,
                "final_relative_residual": final_residual,
                "failure_reason": failure_reason,
            }
        )
    return pl.DataFrame(rows, schema=SOLVE_ATTEMPT_SCHEMA)


def _memory_fields(memory: PeakMemory) -> tuple[int | None, str]:
    match memory:
        case MeasuredPeakMemory(bytes=peak_memory_bytes):
            return peak_memory_bytes, "measured"
        case UnknownPeakMemory():
            return None, "unknown"
        case unreachable:
            assert_never(unreachable)


def _setup_outcome_fields(outcome: SetupOutcome) -> tuple[str, str | None]:
    match outcome:
        case SetupCompleted():
            return "completed", None
        case SetupFailed(reason=reason):
            return "failed", reason
        case unreachable:
            assert_never(unreachable)


def _solve_outcome_fields(outcome: SolveOutcome) -> tuple[str, int, float | None, str | None]:
    match outcome:
        case SolveConverged(iterations=iterations, final_relative_residual=residual):
            return "converged", iterations, residual, None
        case SolveFailed(iterations=iterations, reason=reason):
            return "failed", iterations, None, reason
        case unreachable:
            assert_never(unreachable)
