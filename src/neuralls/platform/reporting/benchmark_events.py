"""Versioned persistence for canonical benchmark setup and solve events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import polars as pl
import pyarrow.parquet as pq
import tomli_w

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    PeakMemory,
    SolveAttempt,
    SystemTag,
)
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.workload import (
    IncompleteWorkloadSummary,
    MethodWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.event_frames import (
    SETUP_EVENT_SCHEMA,
    SOLVE_ATTEMPT_SCHEMA,
    build_setup_frame,
    build_solve_frame,
    validate_setup_solve_frame_links,
)
from neuralls.shared.digest import Digest
from neuralls.shared.types import PreconditionerName

BENCHMARK_EVENT_SCHEMA_VERSION = 1
_SCHEMA_METADATA_KEY = b"neuralls.benchmark_event_schema_version"
_PUBLIC_SCHEMA_METADATA_KEY = b"schema_version"


class CompletedMethodWorkload(Protocol):
    """Structural input contract for the platform serializer.

    The concrete orchestration DTO remains in composition; this boundary only
    requires its persisted fields and therefore does not create an upward
    platform-to-composition dependency.
    """

    method: PreconditionerName
    planned: tuple[SystemTag, ...]
    setup_events: tuple[MatrixSetupEvent, ...]
    solve_attempts: tuple[SolveAttempt, ...]
    generation_cost: StageCost | None
    workload_peak_memory: PeakMemory
    config_digest: Digest


@dataclass(frozen=True, slots=True)
class MethodBenchmarkArtifacts:
    """Stable artifact-relative paths emitted for one method workload."""

    setup_events: Path = Path("setup_events.parquet")
    solve_attempts: Path = Path("solve_attempts.parquet")
    summary: Path = Path("summary.toml")


def write_method_benchmark_artifacts(
    workload: CompletedMethodWorkload, summary: MethodWorkloadSummary, root: Path
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
    _write_summary(workload, canonical_summary, root / artifacts.summary)
    return artifacts


def read_setup_events(path: Path) -> pl.DataFrame:
    """Read and schema-check a setup-event Parquet artifact."""
    frame = _read_frame(path, SETUP_EVENT_SCHEMA)
    _validate_unique_setup_ids(frame)
    return frame


def read_solve_attempts(path: Path, *, setup_events_path: Path | None = None) -> pl.DataFrame:
    """Read solve attempts and optionally validate each setup-event link."""
    frame = _read_frame(path, SOLVE_ATTEMPT_SCHEMA)
    setup_path = setup_events_path or path.with_name("setup_events.parquet")
    if setup_events_path is not None or setup_path.exists():
        setup_frame = read_setup_events(setup_path)
        validate_setup_solve_frame_links(setup_frame, frame)
    return frame


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    """Write a frame with the schema version in Parquet metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    metadata = dict(table.schema.metadata or {})
    version = str(BENCHMARK_EVENT_SCHEMA_VERSION).encode("ascii")
    metadata[_SCHEMA_METADATA_KEY] = version
    metadata[_PUBLIC_SCHEMA_METADATA_KEY] = version
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _read_frame(path: Path, schema: dict[str, pl.DataType | type[pl.DataType]]) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = pq.read_schema(path).metadata or {}
    raw_version = metadata.get(_SCHEMA_METADATA_KEY, metadata.get(_PUBLIC_SCHEMA_METADATA_KEY))
    if raw_version is None:
        raise ValueError(f"missing benchmark event schema version in {path}")
    try:
        version = int(raw_version.decode("ascii"))
    except ValueError as exc:
        raise ValueError(f"invalid benchmark event schema version in {path}") from exc
    if version != BENCHMARK_EVENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark event schema version: {version}")
    frame = pl.read_parquet(path)
    expected = pl.Schema(schema)
    if frame.schema != expected:
        raise ValueError(f"unexpected benchmark event schema in {path}: {frame.schema}")
    return frame


def _validate_unique_setup_ids(frame: pl.DataFrame) -> None:
    ids = frame.get_column("setup_event_id").to_list()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate setup event ID")


def _write_summary(
    workload: CompletedMethodWorkload, summary: MethodWorkloadSummary, path: Path
) -> None:
    peak_bytes, peak_status = _memory_fields(summary.workload_peak_memory)
    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_EVENT_SCHEMA_VERSION,
        "method": {
            "name": str(workload.method),
            "config_digest": workload.config_digest,
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
