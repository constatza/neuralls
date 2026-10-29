"""End-to-end comparison failure-reporting and recovery coverage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient
from torchalg.preconditioners.base import Preconditioner

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison._preconditioner_setup import PreconditionerService
from neuralls.composition.comparison.models import ComparisonCompleted, ComparisonParams
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    MLflowGroupReportSink,
)
from tests.workflows.comparison_support import make_settings, write_dataset_config

pytestmark = pytest.mark.deferred_io("mlflow-run-tracking")


def _artifact_names(client: MlflowClient, run_id: str, path: str) -> set[str]:
    return {Path(item.path).name for item in client.list_artifacts(run_id, path=path)}


def _write_case(
    path: Path,
    *,
    tracking_uri: str,
    experiment: str,
    max_iterations: int = 20,
) -> None:
    payload = {
        "mlflow": {"tracking_uri": tracking_uri},
        "names": {"comparison": experiment},
        "comparison_baseline": {
            "rtol": 1e-6,
            "atol": 1e-14,
            "max_iterations": max_iterations,
            "stopping_criterion": "residual_norm",
            "m_max": 5,
            "normalize_system": "none",
            "reference_preconditioner": "jacobi",
            "preconditioners": [
                {"name": "identity", "type": "identity"},
                {"name": "jacobi", "type": "jacobi"},
            ],
        },
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparison_sweeps": [
            {
                "id_prefix": "task12",
                "matrix_dataset": "solutions",
                "rhs_source": {"kind": "gaussian"},
                "rhs_count": 1,
                "seed": 42,
                "selector": {"kind": "first_n", "n": 2},
            }
        ],
    }
    import tomli_w

    path.write_bytes(tomli_w.dumps(payload).encode("utf-8"))


@pytest.mark.integration
@pytest.mark.usefixtures("stub_input_gate")
@pytest.mark.figure_placeholders
def test_failed_report_finalization_is_retried_on_the_next_call(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finished method children do not masquerade as a completed report session."""
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "solutions.toml", "solutions")
    case_path = tmp_path / "report-recovery.toml"
    _write_case(case_path, tracking_uri=mlflow_tracking_uri, experiment=mlflow_experiment)
    original_finalize = MLflowGroupReportSink.finalize
    finalizations = 0

    def fail_once(sink: MLflowGroupReportSink) -> None:
        nonlocal finalizations
        finalizations += 1
        if finalizations == 1:
            raise RuntimeError("injected report finalization failure")
        original_finalize(sink)

    monkeypatch.setattr(MLflowGroupReportSink, "finalize", fail_once)

    with pytest.raises(RuntimeError):
        run_comparison_batch(case_path, ComparisonParams(), settings)
    recovered = run_comparison_batch(case_path, ComparisonParams(), settings)

    assert recovered and all(isinstance(outcome, ComparisonCompleted) for outcome in recovered)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    assert experiment is not None
    sessions = client.search_runs(
        [experiment.experiment_id], filter_string="tags.phase = 'session_comparison'"
    )
    assert sorted(run.info.status for run in sessions) == ["FAILED", "FINISHED"]
    failed = next(run for run in sessions if run.info.status == "FAILED")
    failed_children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{failed.info.run_id}'",
    )
    assert failed_children
    assert all(run.info.status == "FINISHED" for run in failed_children)
    assert _artifact_names(client, failed.info.run_id, "failure") == {
        "context.json",
        "traceback.txt",
    }
    assert failed.data.tags["failure.phase"] == "finalization"
    finished = next(run for run in sessions if run.info.status == "FINISHED")
    assert "figure_manifest.toml" in _artifact_names(
        client, finished.info.run_id, GROUP_REPORT_ARTIFACT_DIR
    )


@pytest.mark.integration
@pytest.mark.usefixtures("stub_input_gate")
def test_blocking_matrix_load_failure_marks_child_and_parent_with_exact_diagnostics(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "solutions.toml", "solutions")
    case_path = tmp_path / "matrix-load-failure.toml"
    _write_case(case_path, tracking_uri=mlflow_tracking_uri, experiment=mlflow_experiment)
    original = RuntimeError("injected matrix input failure")

    def fail_matrix_load(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise original

    monkeypatch.setattr(comparison_batch, "_load_one_matrix_workload", fail_matrix_load)

    with pytest.raises(RuntimeError) as raised:
        run_comparison_batch(case_path, ComparisonParams(), settings)

    assert raised.value is original
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    parent = next(run for run in runs if "mlflow.parentRunId" not in run.data.tags)
    child = next(
        run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    )
    assert parent.info.status == "FAILED"
    assert child.info.status == "FAILED"
    assert child.data.tags["failure.phase"] == "input_loading"
    assert child.data.tags["failure.scope"] == "matrix"
    assert child.data.tags["failure.method"] == "identity"
    assert child.data.tags["failure.dataset"] == "solutions"
    assert "failure.matrix_id" in child.data.tags
    assert parent.data.tags["failure.phase"] == "method_workloads"
    assert _artifact_names(client, child.info.run_id, "failure") == {
        "context.json",
        "traceback.txt",
    }
    assert _artifact_names(client, parent.info.run_id, "failure") == {
        "context.json",
        "traceback.txt",
    }
    assert not _artifact_names(client, parent.info.run_id, GROUP_REPORT_ARTIFACT_DIR)


@pytest.mark.integration
@pytest.mark.usefixtures("stub_input_gate")
@pytest.mark.figure_placeholders
def test_recoverable_setup_failure_warns_persists_and_allows_later_method(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "solutions.toml", "solutions")
    case_path = tmp_path / "recoverable-setup.toml"
    _write_case(case_path, tracking_uri=mlflow_tracking_uri, experiment=mlflow_experiment)
    original_build = comparison_run._build_scheduled

    def fail_identity(
        service: PreconditionerService,
        matrix: torch.Tensor,
        config: PreconditionerConfig,
    ) -> dict[str, Preconditioner]:
        if config.name == "identity":
            raise RuntimeError("injected recoverable setup failure")
        return original_build(service, matrix, config)

    monkeypatch.setattr(comparison_run, "_build_scheduled", fail_identity)
    outcomes = run_comparison_batch(case_path, ComparisonParams(), settings)

    assert outcomes and all(isinstance(outcome, ComparisonCompleted) for outcome in outcomes)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    assert experiment is not None
    children = client.search_runs(
        [experiment.experiment_id], filter_string="tags.phase = 'comparison_method'"
    )
    assert {run.info.status for run in children} == {"FINISHED"}
    by_method = {run.data.tags["method_id"]: run for run in children}
    assert by_method["identity"].data.metrics["n_setup_failed"] == 2.0
    assert by_method["identity"].data.tags["workload_outcome"] == "completed_with_failures"
    assert by_method["jacobi"].data.metrics["n_setup_failed"] == 0.0
    assert by_method["jacobi"].data.metrics["n_attempted_rhs"] == 2.0
