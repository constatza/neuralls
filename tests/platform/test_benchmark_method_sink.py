"""Method-workload MLflow ownership and artifact identity."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import SOLVER_CONTEXT
from tests.recipe_support import GAUSSIAN

from neuralls.composition.comparison.models import CompletedMethodWorkload, canonical_method_config
from neuralls.composition.tracking.run_specs import build_benchmark_method_run_spec
from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    PreparedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
)
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
    read_presentation,
)
from neuralls.platform.tracking.method_sink import MLflowBenchmarkMethodSink
from neuralls.shared.scientific_description.catalog import REALIZED_LEVELS, make_fact
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId

CONVERGED_ITERATIONS = 2
METHOD_DIR = "method_workload"
EXPECTED_METRIC_KEYS = [
    "n_planned_rhs",
    "n_attempted_rhs",
    "n_converged",
    "success_rate",
    "setup_total_s",
    "solve_total_s",
    "workload_online_s",
    "n_setup_failed",
    "n_solve_failed",
    "iterations_total",
    "online_s_per_attempted_rhs",
    "workload_peak_memory_bytes",
]


@dataclass(slots=True)
class LoggedRun:
    """Metrics and ``(artifact_path, file)`` pairs captured from the sink."""

    metrics: dict[str, float]
    artifacts: list[tuple[str | None, Path]]


@pytest.fixture
def method_workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    method = PreconditionerName("jacobi")
    planned = tuple(
        SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(0), i, SystemId(f"s{i}")) for i in range(2)
    )
    setup = MatrixSetupEvent(
        SetupEventId("setup-0"),
        DatasetId("ds"),
        MatrixId(0),
        method,
        1.0,
        (1.0,),
        MeasuredPeakMemory(1),
        SetupCompleted(),
    )
    attempts = tuple(
        SolveAttempt(
            SetupEventId("setup-0"),
            tag.system_id,
            tag.dataset,
            tag.recipe,
            tag.matrix_id,
            tag.rhs_ordinal,
            method,
            2.0 + tag.rhs_ordinal,
            (2.0 + tag.rhs_ordinal,),
            MeasuredPeakMemory(1),
            SolveConverged(CONVERGED_ITERATIONS, 1e-8),
        )
        for tag in planned
    )
    return CompletedMethodWorkload(
        method=method,
        planned=planned,
        setup_events=(setup,),
        solve_attempts=attempts,
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(2),
        config_digest="sha256:" + "a" * 64,
        method_config={"name": str(method)},
        presentation=ExecutedMethod(prepared_method(str(method)), ()),
    )


@pytest.fixture
def solver_contexts(
    method_workload: CompletedMethodWorkload,
) -> Mapping[SystemId, SolverTimingContext]:
    """The one solver context every planned system was timed under."""
    return {tag.system_id: SOLVER_CONTEXT for tag in method_workload.planned}


@pytest.fixture
def references(method_workload: CompletedMethodWorkload) -> Mapping[SystemId, ReferenceSelection]:
    """No reference is configured for any planned system."""
    return {tag.system_id: NoReferenceConfigured() for tag in method_workload.planned}


@pytest.fixture
def method_sink(
    notation: NotationPolicy,
    group_datasets: dict[DatasetId, DatasetPresentation],
    solver_contexts: Mapping[SystemId, SolverTimingContext],
    references: Mapping[SystemId, ReferenceSelection],
) -> MLflowBenchmarkMethodSink:
    """Sink wired to the fixture dataset presentations and per-system provenance."""
    return MLflowBenchmarkMethodSink(group_datasets, solver_contexts, references)


@pytest.fixture
def logged_run(monkeypatch: pytest.MonkeyPatch) -> LoggedRun:
    """Capture everything the sink logs without opening an MLflow run."""
    run = LoggedRun({}, [])
    prefix = "neuralls.platform.tracking.method_sink.mlflow"
    monkeypatch.setattr(
        f"{prefix}.start_run",
        lambda *args, **kwargs: pytest.fail("method sink must not open MLflow runs"),
    )
    monkeypatch.setattr(f"{prefix}.log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(f"{prefix}.set_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        f"{prefix}.log_metric", lambda key, value, **kwargs: run.metrics.__setitem__(key, value)
    )
    monkeypatch.setattr(
        f"{prefix}.log_artifact",
        lambda path, artifact_path=None, **kwargs: run.artifacts.append(
            (artifact_path, Path(path))
        ),
    )
    return run


@pytest.fixture
def method_summary(method_workload: CompletedMethodWorkload) -> CompleteWorkloadSummary:
    summary = summarize_method_workload(
        method_workload.method,
        method_workload.planned,
        method_workload.setup_events,
        method_workload.solve_attempts,
        method_workload.workload_peak_memory,
    )
    assert isinstance(summary, CompleteWorkloadSummary)
    return summary


def test_method_sink_records_aggregates_without_opening_a_run(
    tmp_path: Path,
    method_sink: MLflowBenchmarkMethodSink,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
    logged_run: LoggedRun,
) -> None:
    method_sink.record_method(method_workload, method_summary, root=tmp_path)

    assert list(logged_run.metrics) == EXPECTED_METRIC_KEYS


def test_method_sink_logs_total_iterations_of_every_attempt(
    tmp_path: Path,
    method_sink: MLflowBenchmarkMethodSink,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
    logged_run: LoggedRun,
) -> None:
    method_sink.record_method(method_workload, method_summary, root=tmp_path)

    assert logged_run.metrics["iterations_total"] == float(
        CONVERGED_ITERATIONS * len(method_workload.solve_attempts)
    )


def test_method_sink_logs_exactly_the_artifact_set_and_no_figures(
    tmp_path: Path,
    method_sink: MLflowBenchmarkMethodSink,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
    logged_run: LoggedRun,
) -> None:
    method_sink.record_method(method_workload, method_summary, root=tmp_path)

    tables = {path.name for where, path in logged_run.artifacts if where == METHOD_DIR}
    assert tables == set(MethodBenchmarkArtifacts.file_names())
    assert len(logged_run.artifacts) == len(tables)


def test_realized_facts_are_a_per_system_table_not_a_run_wide_scalar(
    tmp_path: Path,
    method_sink: MLflowBenchmarkMethodSink,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
    logged_run: LoggedRun,
) -> None:
    """A fact that differs between matrices must never be flattened into one parameter."""
    systems = tuple(
        SystemRealization(
            SystemId(f"s{index}"),
            MatrixId(index),
            (make_fact(REALIZED_LEVELS, 3 + index),),
        )
        for index in range(2)
    )
    workload = replace(
        method_workload,
        presentation=ExecutedMethod(method_workload.presentation.prepared, systems),
    )

    method_sink.record_method(workload, method_summary, root=tmp_path)

    logged = {path.name: path for _, path in logged_run.artifacts}
    stored = read_presentation(logged["presentation.toml"])
    assert stored.method.systems == systems


def test_method_sink_marks_recoverable_setup_failures_from_canonical_events(
    monkeypatch: pytest.MonkeyPatch,
    method_workload: CompletedMethodWorkload,
) -> None:
    failed_setup = replace(method_workload.setup_events[0], outcome=SetupFailed("setup broke"))
    failed_workload = replace(
        method_workload,
        setup_events=(failed_setup,),
        solve_attempts=(),
    )
    summary = summarize_method_workload(
        failed_workload.method,
        failed_workload.planned,
        failed_workload.setup_events,
        failed_workload.solve_attempts,
        failed_workload.workload_peak_memory,
    )
    assert isinstance(summary, IncompleteWorkloadSummary)
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.log_metric",
        lambda key, value, **kwargs: metrics.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.set_tag",
        lambda key, value: tags.__setitem__(key, value),
    )

    MLflowBenchmarkMethodSink._log_method_metadata(failed_workload, summary)

    assert metrics["n_setup_failed"] == 1.0
    assert metrics["n_solve_failed"] == 0.0
    assert tags["workload_outcome"] == "completed_with_failures"


def test_method_sink_counts_recoverable_solve_failures_from_canonical_events(
    monkeypatch: pytest.MonkeyPatch,
    method_workload: CompletedMethodWorkload,
) -> None:
    failed_attempt = replace(
        method_workload.solve_attempts[0],
        outcome=SolveFailed(iterations=2, reason="solver broke"),
    )
    failed_workload = replace(
        method_workload,
        solve_attempts=(failed_attempt, *method_workload.solve_attempts[1:]),
    )
    summary = summarize_method_workload(
        failed_workload.method,
        failed_workload.planned,
        failed_workload.setup_events,
        failed_workload.solve_attempts,
        failed_workload.workload_peak_memory,
    )
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.log_metric",
        lambda key, value, **kwargs: metrics.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.method_sink.mlflow.set_tag",
        lambda key, value: tags.__setitem__(key, value),
    )

    MLflowBenchmarkMethodSink._log_method_metadata(failed_workload, summary)

    assert metrics["n_setup_failed"] == 0.0
    assert metrics["n_solve_failed"] == 1.0
    assert tags["workload_outcome"] == "completed_with_failures"


def test_method_child_tags_include_identity_and_required_raw_artifacts(
    prepared_method: PreparedMethodFactory,
) -> None:
    _, tags = build_benchmark_method_run_spec(
        prepared=prepared_method("jacobi"),
        notation=NotationPolicy.default(),
        config_digest="sha256:" + "a" * 64,
        system_ids=("s0", "s1"),
    )
    values = tags.as_mlflow_tags()

    assert values["phase"] == "comparison_method"
    assert "benchmark_event_schema_version" not in values
    assert values["method_config_digest"] == "sha256:" + "a" * 64
    assert values["workload_membership_digest"].startswith("sha256:")
    assert "setup_events.parquet" in values["required_artifacts"]
    assert "method_config.toml" in values["required_artifacts"]
    assert json.loads(values["identity_components"]) == {
        "method_config": "sha256:" + "a" * 16,
        "workload_membership": values["workload_membership_digest"][:23],
    }
    assert values["neuralls.identity.stage"] == "comparison_method"


def _identity_components(
    prepared: PreparedMethodPresentation, config_digest: str, system_ids: tuple[str, ...]
) -> dict[str, str]:
    _, tags = build_benchmark_method_run_spec(
        prepared=prepared,
        notation=NotationPolicy.default(),
        config_digest=config_digest,
        system_ids=system_ids,
    )
    return json.loads(tags.identity_components)


def test_method_identity_has_only_real_inputs_and_is_deterministic_and_sensitive(
    prepared_method: PreparedMethodFactory,
) -> None:
    config = "sha256:" + "a" * 64
    jacobi = prepared_method("jacobi")
    base = _identity_components(jacobi, config, ("s0", "s1"))

    assert set(base) == {"method_config", "workload_membership"}
    assert base == _identity_components(jacobi, config, ("s0", "s1"))
    assert (
        base["method_config"]
        != _identity_components(jacobi, "sha256:" + "b" * 64, ("s0", "s1"))["method_config"]
    )
    assert (
        base["workload_membership"]
        != _identity_components(jacobi, config, ("s0", "s2"))["workload_membership"]
    )


def test_method_membership_digest_includes_ordered_system_identities(
    prepared_method: PreparedMethodFactory,
) -> None:
    """Changing solver/data identity cannot reuse the same visible system IDs."""
    _, first = build_benchmark_method_run_spec(
        prepared=prepared_method("jacobi"),
        notation=NotationPolicy.default(),
        config_digest="sha256:" + "a" * 64,
        system_ids=("s0", "s1"),
        system_identities=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
    )
    _, changed = build_benchmark_method_run_spec(
        prepared=prepared_method("jacobi"),
        notation=NotationPolicy.default(),
        config_digest="sha256:" + "a" * 64,
        system_ids=("s0", "s1"),
        system_identities=("sha256:" + "1" * 64, "sha256:" + "3" * 64),
    )

    assert first.workload_membership_digest != changed.workload_membership_digest


@pytest.mark.parametrize("config_kind", ("amg", "checkpoint"))
def test_method_config_serialization_is_canonical_and_omits_unset_optionals(
    tmp_path: Path,
    config_kind: str,
    method_sink: MLflowBenchmarkMethodSink,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
    logged_run: LoggedRun,
) -> None:
    """Supported configs with unset optional values remain TOML serializable."""
    if config_kind == "amg":
        config = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    else:
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"weights")
        config = NeuralPreconditionerConfig(checkpoint_path=checkpoint)
    expected = canonical_method_config(config)

    workload = replace(method_workload, method_config=expected)

    method_sink.record_method(workload, method_summary, root=tmp_path / "out")

    logged = {path.name: path for _, path in logged_run.artifacts}
    assert tomllib.loads(logged["method_config.toml"].read_text(encoding="utf-8")) == expected
