from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import polars as pl
import pyarrow.parquet as pq
import pytest
from tests.convergence_support import cg_result
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import datasets_for, provenance_for
from tests.realized_support import completed_workload
from tests.recipe_support import GAUSSIAN

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
    ReferenceUnavailable,
)
from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    ExecutedMethod,
    MethodPresentationCandidate,
    MissingMethodPresentationError,
    PreparedMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
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
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary, MethodWorkloadSummary
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.event_frames import (
    build_planned_systems_frame,
    decode_planned_systems_frame,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
    StoredMethodPresentation,
    read_convergence,
    read_planned_systems,
    read_presentation,
    read_setup_events,
    read_solve_attempts,
    write_method_benchmark_artifacts,
)
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.types import (
    CostProvenance,
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)

ARTIFACT_FILE_NAMES = (
    "setup_events.parquet",
    "solve_attempts.parquet",
    "convergence.parquet",
    "planned_systems.parquet",
    "summary.toml",
    "method_config.toml",
    "presentation.toml",
)
CONFIGURED_REFERENCE = ConfiguredReference(PreconditionerName("ic0"))
UNAVAILABLE_REFERENCE = ReferenceUnavailable(PreconditionerName("ic0"), "factorization failed")
UNKNOWN_KIND = "mystery"
REALIZED_DEPTHS = {0: 2, 1: 3}
OTHER_TAG = SystemTag(DatasetId("dataset"), GAUSSIAN, MatrixId(0), 0, SystemId("other"))


class WriteArtifacts(Protocol):
    """Writer bound to default provenance and the shared dataset presentation."""

    def __call__(
        self,
        workload: CompletedMethodWorkload,
        summary: MethodWorkloadSummary,
        root: Path,
    ) -> MethodBenchmarkArtifacts: ...


def _rewrite_frame(path: Path, frame: pl.DataFrame) -> None:
    """Rewrite a test artifact while preserving its Parquet metadata."""
    metadata = pq.read_schema(path).metadata
    pq.write_table(frame.to_arrow().replace_schema_metadata(metadata), path)


@pytest.fixture
def write(dataset_presentation: DatasetPresentation) -> WriteArtifacts:
    """The artifact writer with no-reference provenance for every planned system."""

    def _write(
        workload: CompletedMethodWorkload, summary: MethodWorkloadSummary, root: Path
    ) -> MethodBenchmarkArtifacts:
        return write_method_benchmark_artifacts(
            workload,
            summary,
            root,
            systems=provenance_for(workload.planned),
            datasets=datasets_for(workload.planned, dataset_presentation),
        )

    return _write


@pytest.fixture
def workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    method = PreconditionerName("jacobi")
    planned = (
        SystemTag(DatasetId("dataset"), GAUSSIAN, MatrixId(0), 0, SystemId("s0")),
        SystemTag(DatasetId("dataset"), GAUSSIAN, MatrixId(0), 1, SystemId("s1")),
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
            GAUSSIAN,
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
            GAUSSIAN,
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
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(512),
        config_digest="sha256:" + "a" * 64,
        method_config={"name": str(method)},
        presentation=ExecutedMethod(prepared_method(str(method)), ()),
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
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write(workload=workload, summary=summary, root=tmp_path)

    assert artifacts.setup_events == Path("setup_events.parquet")
    assert artifacts.solve_attempts == Path("solve_attempts.parquet")
    assert artifacts.summary == Path("summary.toml")
    setup = read_setup_events(tmp_path / artifacts.setup_events)
    solves = read_solve_attempts(tmp_path / artifacts.solve_attempts)

    assert setup.schema == pl.Schema(
        {
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
    )
    assert solves.schema == pl.Schema(
        {
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
    )
    assert setup.to_dicts() == [
        {
            "setup_event_id": "setup-0",
            "dataset": "dataset",
            "matrix_id": 0,
            "method": "jacobi",
            "setup_s": 1.25,
            "samples_s": [1.0, 1.5, 1.25],
            "peak_memory_bytes": 128,
            "memory_status": "measured",
            "setup_status": "completed",
            "failure_reason": None,
        }
    ]
    assert solves.to_dicts() == [
        {
            "setup_event_id": "setup-0",
            "system_id": "s0",
            "dataset": "dataset",
            "recipe": recipe_key(GAUSSIAN),
            "matrix_id": 0,
            "rhs_ordinal": 0,
            "method": "jacobi",
            "solve_s": 2.5,
            "samples_s": [2.0, 2.5, 3.0],
            "peak_memory_bytes": None,
            "memory_status": "unknown",
            "solve_status": "converged",
            "iterations": 4,
            "final_relative_residual": 1e-8,
            "failure_reason": None,
        },
        {
            "setup_event_id": "setup-0",
            "system_id": "s1",
            "dataset": "dataset",
            "recipe": recipe_key(GAUSSIAN),
            "matrix_id": 0,
            "rhs_ordinal": 1,
            "method": "jacobi",
            "solve_s": 0.75,
            "samples_s": [0.75],
            "peak_memory_bytes": 256,
            "memory_status": "measured",
            "solve_status": "failed",
            "iterations": 7,
            "final_relative_residual": None,
            "failure_reason": "breakdown",
        },
    ]


def test_event_reader_rejects_schema_valid_invalid_measurements(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write(workload=workload, summary=summary, root=tmp_path)
    setup_path = tmp_path / artifacts.setup_events
    setup = read_setup_events(setup_path).with_columns(pl.lit(-1.0).alias("setup_s"))
    _rewrite_frame(setup_path, setup)

    with pytest.raises(ValueError):
        read_setup_events(setup_path)


def test_summary_is_reconciled_from_canonical_frames_and_versioned(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write(workload=workload, summary=summary, root=tmp_path)
    setup = read_setup_events(tmp_path / artifacts.setup_events)
    solves = read_solve_attempts(tmp_path / artifacts.solve_attempts)
    payload = tomllib.loads((tmp_path / artifacts.summary).read_text(encoding="utf-8"))

    setup_total = sum(setup.get_column("setup_s").to_list())
    solve_total = sum(solves.get_column("solve_s").to_list())
    attempted = solves.height
    converged = solves.filter(pl.col("solve_status") == "converged").height
    measured_peaks = [
        value
        for value in (
            *setup.get_column("peak_memory_bytes").to_list(),
            *solves.get_column("peak_memory_bytes").to_list(),
        )
        if value is not None
    ]
    workload_total = setup_total + solve_total

    assert "schema_version" not in payload
    assert payload["method"] == {"name": "jacobi", "config_digest": "sha256:" + "a" * 64}
    assert payload["counts"] == {
        "planned_rhs": attempted,
        "attempted_rhs": attempted,
        "converged": converged,
    }
    assert payload["online"] == {
        "setup_s": setup_total,
        "solve_s": solve_total,
        "workload_s": workload_total,
        "per_attempted_rhs_s": workload_total / attempted,
    }
    assert max(measured_peaks) == 256
    assert isinstance(summary.workload_peak_memory, MeasuredPeakMemory)
    assert payload["memory"] == {
        "peak_bytes": summary.workload_peak_memory.bytes,
        "status": "measured",
    }
    assert "generation_s" not in payload


def test_unknown_workload_memory_writes_status_without_toml_null(
    write: WriteArtifacts, tmp_path: Path, workload: CompletedMethodWorkload
) -> None:
    unknown_workload = replace(workload, workload_peak_memory=UnknownPeakMemory())
    unknown_summary = summarize_method_workload(
        unknown_workload.method,
        unknown_workload.planned,
        unknown_workload.setup_events,
        unknown_workload.solve_attempts,
        unknown_workload.workload_peak_memory,
    )
    artifacts = write(workload=unknown_workload, summary=unknown_summary, root=tmp_path)
    memory = tomllib.loads((tmp_path / artifacts.summary).read_text(encoding="utf-8"))["memory"]
    assert memory == {"status": "unknown"}


def test_generation_cost_is_explicitly_named_and_only_cost_persisted(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    with_generation = replace(
        workload,
        generation_cost=StageCost(8.5, 1024, CostProvenance.HISTORICAL),
    )
    artifacts = write(workload=with_generation, summary=summary, root=tmp_path)
    payload = tomllib.loads((tmp_path / artifacts.summary).read_text(encoding="utf-8"))
    assert payload["generation_s"] == 8.5
    assert payload["generation_source"] == "historical"
    assert "offline" not in payload


@pytest.mark.parametrize("digest", ["", "sha256:short", "sha256:" + "G" * 64])
def test_writer_rejects_empty_or_malformed_config_digest(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
    digest: str,
) -> None:
    with pytest.raises(ValueError):
        write(workload=replace(workload, config_digest=digest), summary=summary, root=tmp_path)


def test_reader_rejects_same_id_identity_mismatch_and_failed_setup(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write(workload, summary, tmp_path / "identity")
    solve_path = tmp_path / "identity" / artifacts.solve_attempts
    solves = pl.read_parquet(solve_path).with_columns(pl.lit("other").alias("dataset"))
    _rewrite_frame(solve_path, solves)
    with pytest.raises(ValueError):
        read_solve_attempts(
            solve_path, setup_events_path=tmp_path / "identity" / artifacts.setup_events
        )

    artifacts = write(workload, summary, tmp_path / "failed")
    setup_path = tmp_path / "failed" / artifacts.setup_events
    setup = pl.read_parquet(setup_path).with_columns(pl.lit("failed").alias("setup_status"))
    _rewrite_frame(setup_path, setup)
    with pytest.raises(ValueError):
        read_solve_attempts(
            tmp_path / "failed" / artifacts.solve_attempts, setup_events_path=setup_path
        )


def test_failed_setup_outcome_round_trips_without_inventing_attempts(
    write: WriteArtifacts, tmp_path: Path, workload: CompletedMethodWorkload
) -> None:
    failed_tag = SystemTag(DatasetId("dataset"), GAUSSIAN, MatrixId(1), 0, SystemId("s2"))
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
    artifacts = write(workload=failed_workload, summary=failed_summary, root=tmp_path)
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
    zero_artifacts = write(
        workload=zero_attempt_workload, summary=zero_attempt_summary, root=zero_root
    )
    zero_payload = tomllib.loads((zero_root / zero_artifacts.summary).read_text(encoding="utf-8"))
    assert "per_attempted_rhs_s" not in zero_payload["online"]


def test_incomplete_summary_preserves_planned_membership_and_missing_ids(
    write: WriteArtifacts, tmp_path: Path, workload: CompletedMethodWorkload
) -> None:
    failed_tag = SystemTag(DatasetId("dataset"), GAUSSIAN, MatrixId(1), 0, SystemId("s2"))
    failed_setup = MatrixSetupEvent(
        SetupEventId("setup-1"),
        DatasetId("dataset"),
        MatrixId(1),
        workload.method,
        0.5,
        (0.5,),
        UnknownPeakMemory(),
        SetupFailed("assembly failed"),
    )
    incomplete_workload = replace(
        workload,
        planned=(*workload.planned, failed_tag),
        setup_events=(*workload.setup_events, failed_setup),
    )
    incomplete_summary = summarize_method_workload(
        incomplete_workload.method,
        incomplete_workload.planned,
        incomplete_workload.setup_events,
        incomplete_workload.solve_attempts,
        incomplete_workload.workload_peak_memory,
    )
    artifacts = write(workload=incomplete_workload, summary=incomplete_summary, root=tmp_path)
    setup = read_setup_events(tmp_path / artifacts.setup_events)
    solves = read_solve_attempts(tmp_path / artifacts.solve_attempts)
    payload = tomllib.loads((tmp_path / artifacts.summary).read_text(encoding="utf-8"))

    # Event frames are canonical for realized events; planned membership and
    # the method-lifetime peak remain canonical in the Task 2 summary.
    assert setup.height == 2
    assert solves.height == 2
    assert solves.filter(pl.col("solve_status") == "converged").height == 1
    assert payload["counts"] == {
        "planned_rhs": 3,
        "attempted_rhs": 2,
        "converged": 1,
        "missing_system_ids": ["s2"],
    }


def test_readers_reject_wrong_column_schema_and_corrupt_links(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    artifacts = write(workload=workload, summary=summary, root=tmp_path)
    solve_path = tmp_path / artifacts.solve_attempts
    solves = pl.read_parquet(solve_path).with_columns(pl.lit("missing").alias("setup_event_id"))
    solve_schema = pq.read_schema(solve_path)
    pq.write_table(
        solves.to_arrow().replace_schema_metadata(solve_schema.metadata),
        solve_path,
    )
    with pytest.raises(ValueError):
        read_solve_attempts(solve_path, setup_events_path=tmp_path / artifacts.setup_events)

    setup_path = tmp_path / artifacts.setup_events
    setup = pl.read_parquet(setup_path)
    pq.write_table(setup.drop("setup_event_id").to_arrow(), setup_path)
    with pytest.raises(ValueError, match="unexpected benchmark event schema"):
        read_setup_events(setup_path)

    pq.write_table(setup.to_arrow().replace_schema_metadata({}), setup_path)
    assert read_setup_events(setup_path).equals(setup)


def test_convergence_table_round_trips_and_rejects_a_duplicated_row(
    write: WriteArtifacts,
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
) -> None:
    with_history = replace(
        workload,
        diagnostics=((SystemId("s0"), cg_result([1.0, 0.1], [1.0, 0.3])),),
    )

    artifacts = write(with_history, summary, tmp_path)
    path = tmp_path / artifacts.convergence

    assert read_convergence(path).rows() == [
        ("jacobi", "s0", 0, 1.0, 1.0),
        ("jacobi", "s0", 1, 0.1, 0.3),
    ]
    _rewrite_frame(path, pl.concat([pl.read_parquet(path)] * 2))
    with pytest.raises(ValueError, match="duplicate convergence row"):
        read_convergence(path)


@pytest.fixture
def planned_frame(workload: CompletedMethodWorkload) -> pl.DataFrame:
    """A valid planned-systems frame to corrupt one column at a time."""
    return build_planned_systems_frame(provenance_for(workload.planned))


@pytest.fixture
def qualified_method(dataset_presentation: DatasetPresentation) -> PreparedMethodPresentation:
    """A dataset-trained method prepared in its own validated cohort."""
    cohort = prepare_presentations(
        UnvalidatedComparisonPresentation(
            methods=(
                MethodPresentationCandidate(
                    PreconditionerName("pod"),
                    DatasetQualifiedMethodPresentation(
                        ScientificDescription("POD-2G"), dataset_presentation
                    ),
                ),
            ),
            scope=ComparisonScopePresentation(dataset_presentation),
        )
    )
    return cohort.methods[0]


@pytest.fixture
def summarize() -> Callable[[CompletedMethodWorkload], MethodWorkloadSummary]:
    """The canonical summary of a workload."""

    def build(workload: CompletedMethodWorkload) -> MethodWorkloadSummary:
        return summarize_method_workload(
            workload.method,
            workload.planned,
            workload.setup_events,
            workload.solve_attempts,
            workload.workload_peak_memory,
        )

    return build


@pytest.mark.parametrize(
    "reference", [NoReferenceConfigured(), CONFIGURED_REFERENCE, UNAVAILABLE_REFERENCE]
)
def test_planned_systems_round_trip_each_reference_kind(
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
    dataset_presentation: DatasetPresentation,
    reference: ReferenceSelection,
) -> None:
    systems = provenance_for(workload.planned, reference)
    artifacts = write_method_benchmark_artifacts(
        workload,
        summary,
        tmp_path,
        systems=systems,
        datasets=datasets_for(workload.planned, dataset_presentation),
    )

    assert read_planned_systems(tmp_path / artifacts.planned_systems) == systems


@pytest.mark.parametrize(
    "columns",
    [
        {"reference_kind": UNKNOWN_KIND},
        {"reference_kind": "none", "reference_method": "ic0"},
        {"reference_kind": "unavailable", "reference_method": "ic0", "reference_reason": None},
    ],
    ids=["unknown-kind", "none-with-method", "unavailable-without-reason"],
)
def test_planned_systems_decoder_rejects_inconsistent_reference_columns(
    planned_frame: pl.DataFrame, columns: dict[str, str | None]
) -> None:
    corrupt = planned_frame.with_columns(
        *(pl.lit(value, dtype=pl.String).alias(name) for name, value in columns.items())
    )

    with pytest.raises(ValueError):
        decode_planned_systems_frame(corrupt)


def test_planned_systems_decoder_rejects_duplicate_system_id(planned_frame: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicate planned system ID"):
        decode_planned_systems_frame(pl.concat([planned_frame.head(1)] * 2))


def test_planned_systems_reader_rejects_malformed_artifact(
    tmp_path: Path, planned_frame: pl.DataFrame
) -> None:
    path = tmp_path / "planned_systems.parquet"
    pq.write_table(
        planned_frame.with_columns(pl.lit(UNKNOWN_KIND).alias("reference_kind")).to_arrow(), path
    )

    with pytest.raises(ValueError, match="unknown reference kind"):
        read_planned_systems(path)


def test_presentation_round_trips_standalone_method_with_realized_facts(
    tmp_path: Path,
    prepared_method: PreparedMethodFactory,
    dataset_presentation: DatasetPresentation,
    summarize: Callable[[CompletedMethodWorkload], MethodWorkloadSummary],
    write: WriteArtifacts,
) -> None:
    realized = completed_workload(prepared_method("jacobi"), REALIZED_DEPTHS)
    artifacts = write(realized, summarize(realized), tmp_path)

    assert read_presentation(tmp_path / artifacts.presentation) == StoredMethodPresentation(
        realized.presentation, datasets_for(realized.planned, dataset_presentation)
    )


def test_presentation_round_trips_dataset_qualified_method(
    tmp_path: Path,
    qualified_method: PreparedMethodPresentation,
    dataset_presentation: DatasetPresentation,
    summarize: Callable[[CompletedMethodWorkload], MethodWorkloadSummary],
    write: WriteArtifacts,
) -> None:
    realized = completed_workload(qualified_method, REALIZED_DEPTHS)
    artifacts = write(realized, summarize(realized), tmp_path)

    assert read_presentation(tmp_path / artifacts.presentation) == StoredMethodPresentation(
        realized.presentation, datasets_for(realized.planned, dataset_presentation)
    )


def test_writer_rejects_systems_that_differ_from_the_plan(
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
    dataset_presentation: DatasetPresentation,
) -> None:
    with pytest.raises(ValueError, match="plan order"):
        write_method_benchmark_artifacts(
            workload,
            summary,
            tmp_path,
            systems=provenance_for((OTHER_TAG, *workload.planned[1:])),
            datasets=datasets_for(workload.planned, dataset_presentation),
        )


def test_writer_rejects_a_planned_dataset_without_presentation(
    tmp_path: Path, workload: CompletedMethodWorkload, summary: CompleteWorkloadSummary
) -> None:
    with pytest.raises(MissingMethodPresentationError):
        write_method_benchmark_artifacts(
            workload,
            summary,
            tmp_path,
            systems=provenance_for(workload.planned),
            datasets={},
        )


def test_artifact_set_names_seven_files_and_the_writer_creates_each(
    tmp_path: Path,
    workload: CompletedMethodWorkload,
    summary: CompleteWorkloadSummary,
    write: WriteArtifacts,
) -> None:
    artifacts = write(workload, summary, tmp_path)

    assert MethodBenchmarkArtifacts.file_names() == ARTIFACT_FILE_NAMES
    assert tuple(str(path) for path in artifacts.paths()) == ARTIFACT_FILE_NAMES
    assert all((tmp_path / name).is_file() for name in ARTIFACT_FILE_NAMES)
