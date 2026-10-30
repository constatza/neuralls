"""Normalized Polars serialization of canonical benchmark events."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from math import isfinite
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
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.shared.rhs_recipe import parse_recipe_key, recipe_key
from neuralls.shared.types import PreconditionerName, SystemId

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

CONVERGENCE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "system_id": pl.String,
    "iteration": pl.Int32,
    "residual_rel": pl.Float64,
    "energy_error_rel": pl.Float64,
}


def build_convergence_frame(
    method: PreconditionerName,
    diagnostics: Sequence[tuple[SystemId, CGComparisonResult]],
) -> pl.DataFrame:
    """One row per iteration of each system's relative residual history.

    The energy error is null for iterations the energy history does not cover,
    including every iteration of a system without a reference solution.
    """
    rows: list[dict[str, object]] = []
    for system_id, result in diagnostics:
        energy = result.error_history_a_rel
        for iteration, residual in enumerate(result.residual_history_rel):
            has_energy = energy is not None and iteration < len(energy)
            rows.append(
                {
                    "method": str(method),
                    "system_id": str(system_id),
                    "iteration": iteration,
                    "residual_rel": float(residual),
                    "energy_error_rel": (
                        float(energy[iteration]) if energy is not None and has_energy else None
                    ),
                }
            )
    return pl.DataFrame(rows, schema=CONVERGENCE_SCHEMA)


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
                "recipe": recipe_key(attempt.recipe),
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


def validate_setup_solve_frame_links(setup_frame: pl.DataFrame, solve_frame: pl.DataFrame) -> None:
    """Validate setup identity and outcome for every linked solve row.

    This is shared by in-memory reporting and platform artifact reuse so a
    schema-valid Parquet row cannot associate a solve with the wrong matrix,
    dataset, method, or an unsuccessful setup.
    """
    setups = setup_frame.to_dicts()
    by_id: dict[str, list[dict[str, object]]] = {}
    for row in setups:
        by_id.setdefault(str(row["setup_event_id"]), []).append(row)
    for row in solve_frame.to_dicts():
        event_id = str(row["setup_event_id"])
        matches = by_id.get(event_id, [])
        if not matches:
            raise ValueError(f"solve attempt references unknown setup event {event_id!r}")
        if len(matches) != 1:
            raise ValueError(f"solve attempt must reference exactly one setup event {event_id!r}")
        setup = matches[0]
        if setup["setup_status"] != "completed":
            raise ValueError(f"solve attempt references failed setup {event_id!r}")
        if any(row[field] != setup[field] for field in ("dataset", "matrix_id", "method")):
            raise ValueError(f"solve attempt/setup identity mismatch for {event_id!r}")


def validate_decoded_setup_frame(frame: pl.DataFrame) -> None:
    """Reapply setup-event invariants after an untrusted frame is decoded."""
    setup_keys: set[tuple[str, int, str]] = set()
    for row in frame.to_dicts():
        event_id = _required_text(row["setup_event_id"], "setup event ID")
        dataset = _required_text(row["dataset"], "dataset")
        method = _required_text(row["method"], "method")
        matrix_id = _required_int(row["matrix_id"], "matrix ID")
        _finite_non_negative(row["setup_s"], "setup time")
        _validate_samples(row["samples_s"])
        _validate_memory(row["peak_memory_bytes"], row["memory_status"])
        status = _required_text(row["setup_status"], "setup status")
        reason = row["failure_reason"]
        if status not in {"completed", "failed"}:
            raise ValueError(f"unknown setup status: {status!r}")
        if status == "failed" and not isinstance(reason, str):
            raise ValueError(f"failed setup {event_id!r} is missing a failure reason")
        if status == "completed" and reason is not None:
            raise ValueError(f"completed setup {event_id!r} has a failure reason")
        key = (dataset, matrix_id, method)
        if key in setup_keys:
            raise ValueError("duplicate setup key for dataset, matrix, and method")
        setup_keys.add(key)


def validate_decoded_solve_frame(frame: pl.DataFrame) -> None:
    """Reapply solve-attempt invariants after an untrusted frame is decoded."""
    seen_systems: set[str] = set()
    for row in frame.to_dicts():
        _required_text(row["setup_event_id"], "setup event ID")
        system_id = _required_text(row["system_id"], "system ID")
        _required_text(row["dataset"], "dataset")
        parse_recipe_key(_required_text(row["recipe"], "recipe"))
        _required_int(row["matrix_id"], "matrix ID")
        _required_int(row["rhs_ordinal"], "RHS ordinal")
        _required_text(row["method"], "method")
        _finite_non_negative(row["solve_s"], "solve time")
        _validate_samples(row["samples_s"])
        _validate_memory(row["peak_memory_bytes"], row["memory_status"])
        status = _required_text(row["solve_status"], "solve status")
        _required_int(row["iterations"], "iterations")
        residual = row["final_relative_residual"]
        reason = row["failure_reason"]
        if status not in {"converged", "failed"}:
            raise ValueError(f"unknown solve status: {status!r}")
        if status == "converged":
            if residual is None:
                raise ValueError(f"converged solve {system_id!r} is missing a residual")
            _finite_non_negative(residual, "final relative residual")
            if reason is not None:
                raise ValueError(f"converged solve {system_id!r} has a failure reason")
        else:
            if residual is not None:
                raise ValueError(f"failed solve {system_id!r} has a final relative residual")
            if not isinstance(reason, str):
                raise ValueError(f"failed solve {system_id!r} is missing a failure reason")
        if system_id in seen_systems:
            raise ValueError(f"duplicate solve system ID: {system_id!r}")
        seen_systems.add(system_id)


def validate_decoded_convergence_frame(frame: pl.DataFrame) -> None:
    """Reapply convergence-row invariants after an untrusted frame is decoded."""
    seen: set[tuple[str, str, int]] = set()
    for row in frame.to_dicts():
        key = (
            _required_text(row["method"], "method"),
            _required_text(row["system_id"], "system ID"),
            _required_int(row["iteration"], "iteration"),
        )
        _finite_non_negative(row["residual_rel"], "relative residual")
        if row["energy_error_rel"] is not None:
            _finite_non_negative(row["energy_error_rel"], "relative energy error")
        if key in seen:
            raise ValueError(f"duplicate convergence row: {key!r}")
        seen.add(key)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(value: object, name: str) -> None:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError(f"{name} must be finite and non-negative")
    if not isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_samples(value: object) -> None:
    if not isinstance(value, list):
        raise TypeError("timing samples must be a list")
    for sample in value:
        _finite_non_negative(sample, "timing sample")


def _validate_memory(value: object, status: object) -> None:
    if status == "unknown":
        if value is not None:
            raise ValueError("unknown memory must not carry peak bytes")
    elif status == "measured":
        _required_int(value, "peak memory bytes")
    else:
        raise ValueError(f"unknown memory status: {status!r}")


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
