"""Method-workload MLflow ownership and artifact identity."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from tests.presentation_support import PreparedMethodFactory

from neuralls.composition.comparison.models import CompletedMethodWorkload, canonical_method_config
from neuralls.composition.tracking.run_specs import build_benchmark_method_run_spec
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
from neuralls.platform.tracking.comparison_tracking import (
    MLflowBenchmarkMethodSink,
    _write_method_config,
)
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId


@pytest.fixture
def method_workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    method = PreconditionerName("jacobi")
    planned = tuple(
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(0), i, SystemId(f"s{i}")) for i in range(2)
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
            SolveConverged(2, 1e-8),
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
        presentation=prepared_method(str(method)),
    )


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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method_workload: CompletedMethodWorkload,
    method_summary: CompleteWorkloadSummary,
) -> None:
    metrics: list[tuple[str, float]] = []
    artifacts: list[str] = []
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.start_run",
        lambda *args, **kwargs: pytest.fail("method sink must not open MLflow runs"),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_metric",
        lambda key, value, **kwargs: metrics.append((key, value)),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_artifact",
        lambda path, **kwargs: artifacts.append(Path(path).name),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.set_tag",
        lambda *args, **kwargs: None,
    )

    sink = MLflowBenchmarkMethodSink()
    sink.record_method(method_workload, method_summary, root=tmp_path)

    assert [key for key, _ in metrics] == [
        "n_planned_rhs",
        "n_attempted_rhs",
        "n_converged",
        "success_rate",
        "setup_total_s",
        "solve_total_s",
        "workload_online_s",
        "n_setup_failed",
        "n_solve_failed",
        "online_s_per_attempted_rhs",
        "workload_peak_memory_bytes",
    ]
    assert set(artifacts) >= {
        "setup_events.parquet",
        "solve_attempts.parquet",
        "summary.toml",
        "method_config.toml",
        "method_diagnostics.json",
        "residual_histories.json",
        "energy_histories.json",
        "residual_history_jacobi.png",
        "energy_history_jacobi.png",
    }


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
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_metric",
        lambda key, value, **kwargs: metrics.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.set_tag",
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
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.log_metric",
        lambda key, value, **kwargs: metrics.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "neuralls.platform.tracking.comparison_tracking.mlflow.set_tag",
        lambda key, value: tags.__setitem__(key, value),
    )

    MLflowBenchmarkMethodSink._log_method_metadata(failed_workload, summary)

    assert metrics["n_setup_failed"] == 0.0
    assert metrics["n_solve_failed"] == 1.0
    assert tags["workload_outcome"] == "completed_with_failures"


def test_method_child_tags_include_identity_and_required_raw_artifacts() -> None:
    _, tags = build_benchmark_method_run_spec(
        method="jacobi",
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
    assert "schema" not in values["identity_components"]
    assert "0000000000000000" not in values["identity_components"]
    assert values["neuralls.identity.stage"] == "comparison_method"


def _identity_components(config_digest: str, system_ids: tuple[str, ...]) -> dict[str, str]:
    _, tags = build_benchmark_method_run_spec(
        method="jacobi", config_digest=config_digest, system_ids=system_ids
    )
    return json.loads(tags.identity_components)


def test_method_identity_has_only_real_inputs_and_is_deterministic_and_sensitive() -> None:
    config = "sha256:" + "a" * 64
    base = _identity_components(config, ("s0", "s1"))

    assert set(base) == {"method_config", "workload_membership"}
    assert base == _identity_components(config, ("s0", "s1"))
    assert (
        base["method_config"]
        != _identity_components("sha256:" + "b" * 64, ("s0", "s1"))["method_config"]
    )
    assert (
        base["workload_membership"]
        != _identity_components(config, ("s0", "s2"))["workload_membership"]
    )


def test_method_membership_digest_includes_ordered_system_identities() -> None:
    """Changing solver/data identity cannot reuse the same visible system IDs."""
    _, first = build_benchmark_method_run_spec(
        method="jacobi",
        config_digest="sha256:" + "a" * 64,
        system_ids=("s0", "s1"),
        system_identities=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
    )
    _, changed = build_benchmark_method_run_spec(
        method="jacobi",
        config_digest="sha256:" + "a" * 64,
        system_ids=("s0", "s1"),
        system_identities=("sha256:" + "1" * 64, "sha256:" + "3" * 64),
    )

    assert first.workload_membership_digest != changed.workload_membership_digest


@pytest.mark.parametrize("config_kind", ("amg", "checkpoint"))
def test_method_config_serialization_is_canonical_and_omits_unset_optionals(
    tmp_path: Path, config_kind: str
) -> None:
    """Supported configs with unset optional values remain TOML serializable."""
    if config_kind == "amg":
        config = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    else:
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"weights")
        config = NeuralPreconditionerConfig(checkpoint_path=checkpoint)
    expected = canonical_method_config(config)

    path = _write_method_config(tmp_path, canonical_method_config(config))

    assert tomllib.loads(path.read_text(encoding="utf-8")) == expected
