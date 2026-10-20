from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.solver.models import (
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
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.reporting.benchmark_events import (
    BENCHMARK_EVENT_SCHEMA_VERSION,
    read_setup_events,
    read_solve_attempts,
    write_method_benchmark_artifacts,
)
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId


@pytest.fixture
def workload() -> CompletedMethodWorkload:
    method = PreconditionerName("jacobi")
    planned = (
        SystemTag(DatasetId("dataset"), "gaussian", MatrixId(0), 0, SystemId("s0")),
        SystemTag(DatasetId("dataset"), "gaussian", MatrixId(0), 1, SystemId("s1")),
    )
    setup = (
        MatrixSetupEvent(
            SetupEventId("setup-0"),
            DatasetId("dataset"),
            MatrixId(0),
            method,
            1.25,
            (1.0, 1.5, 1.25),
            MeasuredPeakMemory(128),
            SetupCompleted(),
        ),
    )
    attempts = (
        SolveAttempt(
            SetupEventId("setup-0"),
            SystemId("s0"),
            DatasetId("dataset"),
            "gaussian",
            MatrixId(0),
            0,
            method,
            2.5,
            (2.0, 2.5, 3.0),
            UnknownPeakMemory(),
            SolveConverged(4, 1e-8),
        ),
        SolveAttempt(
            SetupEventId("setup-0"),
            SystemId("s1"),
            DatasetId("dataset"),
            "gaussian",
            MatrixId(0),
            1,
            method,
            0.75,
            (0.75,),
            MeasuredPeakMemory(256),
            SolveFailed(7, "breakdown"),
        ),
    )
    return CompletedMethodWorkload(
        method=method,
        planned=planned,
        setup_events=setup,
        solve_attempts=attempts,
        diagnostics=(),
        offline_costs=(),
        workload_peak_memory=MeasuredPeakMemory(512),
        config_digest="sha256:" + "a" * 64,
    )


@pytest.fixture
def summary(workload: CompletedMethodWorkload) -> CompleteWorkloadSummary:
    result = summarize_method_workload(
        workload.method,
        workload.planned,
        workload.setup_events,
        workload.solve_attempts,
        workload.workload_peak_memory,
    )
    assert isinstance(result, CompleteWorkloadSummary)
    return result


def test_event_artifacts_round_trip_preserves_schema_values_order_links_and_variants(
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write_method_benchmark_artifacts(workload=workload, summary=summary, root=tmp_path)

    assert artifacts.setup_events == Path("setup_events.parquet")
    assert artifacts.solve_attempts == Path("solve_attempts.parquet")
    assert artifacts.summary == Path("summary.toml")
    setup = read_setup_events(tmp_path / artifacts.setup_events)
    solves = read_solve_attempts(tmp_path / artifacts.solve_attempts)

    assert setup.columns == [
        "setup_event_id",
        "dataset",
        "matrix_id",
        "method",
        "setup_s",
        "samples_s",
        "peak_memory_bytes",
        "memory_status",
        "setup_status",
        "failure_reason",
    ]
    assert solves.columns == [
        "setup_event_id",
        "system_id",
        "dataset",
        "recipe",
        "matrix_id",
        "rhs_ordinal",
        "method",
        "solve_s",
        "samples_s",
        "peak_memory_bytes",
        "memory_status",
        "solve_status",
        "iterations",
        "final_relative_residual",
        "failure_reason",
    ]
    assert setup.get_column("setup_event_id").to_list() == ["setup-0"]
    assert setup.get_column("samples_s").to_list() == [[1.0, 1.5, 1.25]]
    assert setup.get_column("memory_status").to_list() == ["measured"]
    assert solves.get_column("system_id").to_list() == ["s0", "s1"]
    assert solves.get_column("setup_event_id").to_list() == ["setup-0", "setup-0"]
    assert solves.get_column("samples_s").to_list() == [[2.0, 2.5, 3.0], [0.75]]
    assert solves.get_column("solve_status").to_list() == ["converged", "failed"]
    assert solves.get_column("memory_status").to_list() == ["unknown", "measured"]


def test_summary_is_reconciled_from_canonical_frames_and_versioned(
    tmp_path: Path, workload, summary
) -> None:
    artifacts = write_method_benchmark_artifacts(workload=workload, summary=summary, root=tmp_path)
    payload = tomllib.loads((tmp_path / artifacts.summary).read_text(encoding="utf-8"))

    assert payload["schema_version"] == BENCHMARK_EVENT_SCHEMA_VERSION
    assert payload["method"] == {"name": "jacobi", "config_digest": "sha256:" + "a" * 64}
    assert payload["counts"] == {"planned_rhs": 2, "attempted_rhs": 2, "converged": 1}
    assert payload["online"] == {
        "setup_s": 1.25,
        "solve_s": 3.25,
        "workload_s": 4.5,
        "per_attempted_rhs_s": 2.25,
    }
    assert payload["memory"] == {"peak_bytes": 512, "status": "measured"}
    assert "generation_s" not in payload


def test_failed_setup_outcome_round_trips_without_inventing_attempts(
    tmp_path: Path, workload: CompletedMethodWorkload
) -> None:
    failed_tag = SystemTag(DatasetId("dataset"), "gaussian", MatrixId(1), 0, SystemId("s2"))
    failed_setup = MatrixSetupEvent(
        SetupEventId("setup-1"),
        DatasetId("dataset"),
        MatrixId(1),
        workload.method,
        0.5,
        (0.5, 0.75),
        UnknownPeakMemory(),
        SetupFailed("assembly failed"),
    )
    failed_workload = replace(
        workload,
        planned=(*workload.planned, failed_tag),
        setup_events=(*workload.setup_events, failed_setup),
    )
    failed_summary = summarize_method_workload(
        failed_workload.method,
        failed_workload.planned,
        failed_workload.setup_events,
        failed_workload.solve_attempts,
        failed_workload.workload_peak_memory,
    )
    artifacts = write_method_benchmark_artifacts(
        workload=failed_workload, summary=failed_summary, root=tmp_path
    )
    setup = read_setup_events(tmp_path / artifacts.setup_events)
    assert setup.get_column("setup_status").to_list() == ["completed", "failed"]
    assert setup.get_column("failure_reason").to_list() == [None, "assembly failed"]
    assert read_solve_attempts(tmp_path / artifacts.solve_attempts).height == 2

    zero_attempt_workload = replace(
        failed_workload,
        planned=(failed_tag,),
        setup_events=(failed_setup,),
        solve_attempts=(),
    )
    zero_attempt_summary = summarize_method_workload(
        zero_attempt_workload.method,
        zero_attempt_workload.planned,
        zero_attempt_workload.setup_events,
        zero_attempt_workload.solve_attempts,
        zero_attempt_workload.workload_peak_memory,
    )
    zero_root = tmp_path / "zero-attempt"
    zero_artifacts = write_method_benchmark_artifacts(
        workload=zero_attempt_workload, summary=zero_attempt_summary, root=zero_root
    )
    zero_payload = tomllib.loads((zero_root / zero_artifacts.summary).read_text(encoding="utf-8"))
    assert "per_attempted_rhs_s" not in zero_payload["online"]


def test_readers_reject_missing_or_wrong_schema_and_corrupt_links(
    tmp_path: Path, workload: CompletedMethodWorkload, summary: CompleteWorkloadSummary
) -> None:
    artifacts = write_method_benchmark_artifacts(workload=workload, summary=summary, root=tmp_path)
    solve_path = tmp_path / artifacts.solve_attempts
    solves = pl.read_parquet(solve_path).with_columns(pl.lit("missing").alias("setup_event_id"))
    solve_schema = pq.read_schema(solve_path)
    pq.write_table(
        solves.to_arrow().replace_schema_metadata(solve_schema.metadata),
        solve_path,
    )
    with pytest.raises(ValueError, match="setup event"):
        read_solve_attempts(solve_path, setup_events_path=tmp_path / artifacts.setup_events)

    setup_path = tmp_path / artifacts.setup_events
    schema = pq.read_schema(setup_path)
    pq.write_table(
        pq.read_table(setup_path).replace_schema_metadata(
            {**(schema.metadata or {}), b"neuralls.benchmark_event_schema_version": b"999"}
        ),
        setup_path,
    )
    with pytest.raises(ValueError, match="schema version"):
        read_setup_events(setup_path)

    pq.write_table(
        pq.read_table(setup_path).replace_schema_metadata({}),
        setup_path,
    )
    with pytest.raises(ValueError, match="schema version"):
        read_setup_events(setup_path)
