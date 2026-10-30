"""Versioned persistence for canonical benchmark setup and solve events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq
import tomli_w

from neuralls.domain.solver.models.benchmark_events import (
    MeasuredPeakMemory,
    PeakMemory,
)
from neuralls.domain.solver.models.workload import (
    IncompleteWorkloadSummary,
    MethodWorkload,
    MethodWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    SETUP_EVENT_SCHEMA,
    SOLVE_ATTEMPT_SCHEMA,
    build_convergence_frame,
    build_setup_frame,
    build_solve_frame,
    validate_decoded_convergence_frame,
    validate_decoded_setup_frame,
    validate_decoded_solve_frame,
    validate_setup_solve_frame_links,
)
from neuralls.shared.digest import Digest, ordered_workload_digest


@dataclass(frozen=True, slots=True)
class MethodBenchmarkArtifacts:
    """Stable artifact-relative paths emitted for one method workload."""

    setup_events: Path = Path("setup_events.parquet")
    solve_attempts: Path = Path("solve_attempts.parquet")
    convergence: Path = Path("convergence.parquet")
    summary: Path = Path("summary.toml")


def write_method_benchmark_artifacts(
    workload: MethodWorkload, summary: MethodWorkloadSummary, root: Path
) -> MethodBenchmarkArtifacts:
    """Write canonical event frames and a summary derived from their reducers."""
    _validate_config_digest(workload.config_digest)
    canonical_summary = summarize_method_workload(
        workload.method,
        workload.planned,
        workload.setup_events,
        workload.solve_attempts,
        workload.workload_peak_memory,
    )
    if canonical_summary != summary:
        raise ValueError("provided method summary does not reconcile with canonical events")

    artifacts = MethodBenchmarkArtifacts()
    root.mkdir(parents=True, exist_ok=True)
    _write_frame(build_setup_frame(workload.setup_events), root / artifacts.setup_events)
    _write_frame(build_solve_frame(workload.solve_attempts), root / artifacts.solve_attempts)
    _write_frame(
        build_convergence_frame(workload.method, workload.diagnostics),
        root / artifacts.convergence,
    )
    _write_summary(workload, canonical_summary, root / artifacts.summary)
    return artifacts


def read_setup_events(path: Path) -> pl.DataFrame:
    """Read and schema-check a setup-event Parquet artifact."""
    frame = _read_frame(path, SETUP_EVENT_SCHEMA)
    validate_decoded_setup_frame(frame)
    _validate_unique_setup_ids(frame)
    return frame


def read_convergence(path: Path) -> pl.DataFrame:
    """Read and validate a per-iteration convergence Parquet artifact."""
    frame = _read_frame(path, CONVERGENCE_SCHEMA)
    validate_decoded_convergence_frame(frame)
    return frame


def read_solve_attempts(path: Path, *, setup_events_path: Path | None = None) -> pl.DataFrame:
    """Read solve attempts and optionally validate each setup-event link."""
    frame = _read_frame(path, SOLVE_ATTEMPT_SCHEMA)
    validate_decoded_solve_frame(frame)
    setup_path = setup_events_path or path.with_name("setup_events.parquet")
    if setup_events_path is not None or setup_path.exists():
        setup_frame = read_setup_events(setup_path)
        validate_setup_solve_frame_links(setup_frame, frame)
    return frame


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    """Write a frame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(frame.to_arrow(), path)


def _read_frame(path: Path, schema: dict[str, pl.DataType | type[pl.DataType]]) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path)
    expected = pl.Schema(schema)
    if frame.schema != expected:
        raise ValueError(f"unexpected benchmark event schema in {path}: {frame.schema}")
    return frame


def _validate_unique_setup_ids(frame: pl.DataFrame) -> None:
    ids = frame.get_column("setup_event_id").to_list()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate setup event ID")


def _write_summary(workload: MethodWorkload, summary: MethodWorkloadSummary, path: Path) -> None:
    # The reducer's lifetime peak is intentionally not reconstructed from the
    # setup/solve frame maxima; those frames contain only per-event peaks.
    peak_bytes, peak_status = _memory_fields(summary.workload_peak_memory)
    payload: dict[str, Any] = {
        "method": {
            "name": str(workload.method),
            "config_digest": workload.config_digest,
        },
        "workload": {
            "membership_digest": ordered_workload_digest(
                tuple(str(tag.system_id) for tag in workload.planned),
                tuple(getattr(workload, "workload_identities", ())) or None,
            ),
            "system_ids": [str(tag.system_id) for tag in workload.planned],
        },
        "counts": {
            "planned_rhs": summary.n_planned_rhs,
            "attempted_rhs": summary.n_attempted_rhs,
            "converged": summary.n_converged,
        },
        "online": {
            "setup_s": summary.setup_total_s,
            "solve_s": summary.solve_total_s,
            "workload_s": summary.workload_online_s,
        },
        "memory": {"status": peak_status},
    }
    if summary.online_s_per_attempted_rhs is not None:
        payload["online"]["per_attempted_rhs_s"] = summary.online_s_per_attempted_rhs
    if peak_bytes is not None:
        memory = cast(dict[str, Any], dict(payload["memory"]))
        memory["peak_bytes"] = peak_bytes
        payload["memory"] = memory
    if isinstance(summary, IncompleteWorkloadSummary):
        counts = cast(dict[str, Any], dict(payload["counts"]))
        counts["missing_system_ids"] = list(summary.missing_system_ids)
        payload["counts"] = counts
    if workload.generation_cost is not None:
        # Generation is intentionally the only offline cost in this artifact.
        generation = workload.generation_cost
        payload["generation_s"] = generation.wall_time_seconds
        payload["generation_source"] = generation.provenance.value
    path.write_bytes(tomli_w.dumps(payload).encode("utf-8"))


def _memory_fields(memory: PeakMemory) -> tuple[int | None, str]:
    if isinstance(memory, MeasuredPeakMemory):
        return memory.bytes, "measured"
    return None, "unknown"


def _validate_config_digest(digest: Digest) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("config digest must match sha256:<64 lowercase hexadecimal characters>")
