"""Versioned persistence for canonical benchmark setup and solve events."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq
import tomli_w

from neuralls.domain.presentation.methods import ExecutedMethod, MissingMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import (
    MeasuredPeakMemory,
    PeakMemory,
)
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.models.workload import (
    IncompleteWorkloadSummary,
    MethodWorkload,
    MethodWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    PLANNED_SYSTEM_SCHEMA,
    SETUP_EVENT_SCHEMA,
    SOLVE_ATTEMPT_SCHEMA,
    build_convergence_frame,
    build_planned_systems_frame,
    build_setup_frame,
    build_solve_frame,
    decode_planned_systems_frame,
    validate_decoded_convergence_frame,
    validate_decoded_setup_frame,
    validate_decoded_solve_frame,
    validate_setup_solve_frame_links,
)
from neuralls.platform.storage.facts_codec import (
    decode_dataset_presentation,
    decode_executed_method,
    encode_dataset_presentation,
    encode_executed_method,
)
from neuralls.shared.digest import Digest, Json, ordered_workload_digest
from neuralls.shared.types import DatasetId

METHOD_ARTIFACT_DIR = "method_workload"


@dataclass(frozen=True, slots=True)
class MethodBenchmarkArtifacts:
    """The one definition of the artifact set a method workload persists.

    The writer, the reuse gate and the run's ``required_artifacts`` tag all derive
    from these names, so they cannot disagree about what a complete method run holds.
    """

    setup_events: Path = Path("setup_events.parquet")
    solve_attempts: Path = Path("solve_attempts.parquet")
    convergence: Path = Path("convergence.parquet")
    planned_systems: Path = Path("planned_systems.parquet")
    summary: Path = Path("summary.toml")
    method_config: Path = Path("method_config.toml")
    presentation: Path = Path("presentation.toml")

    @classmethod
    def file_names(cls) -> tuple[str, ...]:
        """File names of every artifact of a complete method run, in declaration order."""
        return tuple(str(getattr(cls(), field.name)) for field in fields(cls))

    def paths(self) -> tuple[Path, ...]:
        """Relative paths of every artifact."""
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True, slots=True)
class StoredMethodPresentation:
    """A method's stored presentation and the presentation of each dataset it ran on."""

    method: ExecutedMethod
    datasets: Mapping[DatasetId, DatasetPresentation]


def write_method_benchmark_artifacts(
    workload: MethodWorkload,
    summary: MethodWorkloadSummary,
    root: Path,
    *,
    systems: Sequence[SystemProvenance],
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> MethodBenchmarkArtifacts:
    """Write every artifact of a method workload; the summary is derived from its events.

    ``systems`` carries the solver settings and reference state of each planned system, in
    plan order; ``datasets`` presents each dataset the workload ran on.
    """
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
    if tuple(system.tag for system in systems) != workload.planned:
        raise ValueError("system provenance must describe the planned systems, in plan order")

    artifacts = MethodBenchmarkArtifacts()
    root.mkdir(parents=True, exist_ok=True)
    _write_frame(build_setup_frame(workload.setup_events), root / artifacts.setup_events)
    _write_frame(build_solve_frame(workload.solve_attempts), root / artifacts.solve_attempts)
    _write_frame(
        build_convergence_frame(workload.method, workload.diagnostics),
        root / artifacts.convergence,
    )
    _write_frame(build_planned_systems_frame(systems), root / artifacts.planned_systems)
    _write_summary(workload, canonical_summary, root / artifacts.summary)
    # tomli_w rejects None, so unset model fields are dropped rather than written as null.
    (root / artifacts.method_config).write_bytes(
        tomli_w.dumps(_exclude_none(workload.method_config)).encode("utf-8")
    )
    _write_presentation(workload, datasets, root / artifacts.presentation)
    return artifacts


def read_planned_systems(path: Path) -> tuple[SystemProvenance, ...]:
    """Read and validate a planned-systems Parquet artifact."""
    return decode_planned_systems_frame(_read_frame(path, PLANNED_SYSTEM_SCHEMA))


def read_presentation(path: Path) -> StoredMethodPresentation:
    """Read and validate a presentation TOML artifact."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return StoredMethodPresentation(
        method=decode_executed_method(payload["method"]),
        datasets={
            DatasetId(str(entry["id"])): decode_dataset_presentation(entry["presentation"])
            for entry in payload["datasets"]
        },
    )


def _write_presentation(
    workload: MethodWorkload,
    datasets: Mapping[DatasetId, DatasetPresentation],
    path: Path,
) -> None:
    """Stage the method's presentation with each dataset the workload ran on, as keys and values."""
    ran_on = dict.fromkeys(tag.dataset for tag in workload.planned)
    missing = next((dataset for dataset in ran_on if dataset not in datasets), None)
    if missing is not None:
        raise MissingMethodPresentationError(str(missing))
    payload = {
        "method": encode_executed_method(workload.presentation),
        "datasets": [
            {"id": str(dataset), "presentation": encode_dataset_presentation(datasets[dataset])}
            for dataset in ran_on
        ],
    }
    path.write_bytes(tomli_w.dumps(payload).encode("utf-8"))


def _exclude_none(config: Mapping[str, Json]) -> dict[str, Json]:
    """Recursively remove unset model fields before TOML serialization."""
    return {key: _exclude_none_value(value) for key, value in config.items() if value is not None}


def _exclude_none_value(value: Json) -> Json:
    if isinstance(value, Mapping):
        return _exclude_none(value)
    if isinstance(value, list):
        return [_exclude_none_value(item) for item in value]
    return value


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
