"""Lightweight MLflow artifact-location tests using local SQLite tracking."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
import tomli_w
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._training_artifacts import (
    _log_training_evaluation,
    _normalize_training_numpy_payload,
)
from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.reporting.training_diagnostics import compute_diagnostics
from neuralls.platform.tracking.mlflow_client import log_diagnostics_to_mlflow


def _sqlite_tracking_uri(db_path: Path) -> str:
    """Build sqlite tracking URI from DB path."""
    return build_sqlite_tracking_uri(db_path)


def _ensure_experiment(
    *,
    client: MlflowClient,
    name: str,
    artifact_root: Path,
) -> str:
    """Create experiment if needed and return id."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    try:
        return client.create_experiment(
            name=name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    except MlflowException:
        existing = client.get_experiment_by_name(name)
        if existing is None:
            raise
        return existing.experiment_id


def test_training_logs_diagnostics_artifact_to_mlflow_with_sqlite(tmp_path: Path) -> None:
    """Training diagnostics are logged under figures/ in the active MLflow run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    predictions = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    targets = np.array([[1.1], [1.9], [3.2]], dtype=np.float64)
    contract = default_training_dataset_contract()
    normalized_payload = _normalize_training_numpy_payload(
        {
            "predictions": {"output": predictions.flatten()},
            "targets": {"y": targets.flatten()},
        },
        contract,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-train") as run:
        run_id = run.info.run_id

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id=run_id,
        numpy_payload=normalized_payload,
        metrics_dir=tmp_path / "metrics",
        contract=contract,
    )

    run_data = client.get_run(run_id).data
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics
    assert "eval/rmse" in run_data.metrics


def test_log_diagnostics_to_mlflow_reopens_sqlite_run_without_active_context(
    tmp_path: Path,
) -> None:
    """Diagnostics helper logs directly to an existing SQLite-backed run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-direct-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-direct") as run:
        run_id = run.info.run_id

    diagnostics = compute_diagnostics(
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[1.1], [1.9]], dtype=np.float64),
    )

    log_diagnostics_to_mlflow(tracking_uri, run_id, diagnostics)

    run_data = client.get_run(run_id).data
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics


def _write_dataset_config(path: Path, dataset_id: str, data_dir: Path) -> None:
    """Write a minimal dataset config for the registry."""
    path.write_text(
        "\n".join(
            [
                f'id = "{dataset_id}"',
                "",
                "[output]",
                f'data_dir = "{data_dir.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_experiments_config(
    path: Path,
    output_root: Path,
    *,
    dataset_dir: Path | None = None,
    method_config_path: Path | None = None,
) -> str:
    """Write sqlite-only experiments topology and return tracking URI.

    Args:
        path: Output path for the experiments TOML.
        output_root: Output root directory for MLflow tracking.
        dataset_dir: Optional dataset directory for registry dataset configs.
        method_config_path: Optional method override TOML path for [[comparisons]].

    Returns:
        SQLite tracking URI string.
    """
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    payload: dict[str, object] = {
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
        "names": {
            "training": "neuralls-training",
            "comparison": "Comparisons",
        },
        "comparison_baseline": {
            "rtol": 1e-6,
            "atol": 1e-14,
            "max_iterations": 20,
            "stopping_criterion": "residual_norm",
            "m_max": 5,
            "normalize_system": "matrix",
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
    }
    if dataset_dir is not None:
        datasets_dir = path.parent / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        _write_dataset_config(datasets_dir / "benchmark.toml", "benchmark", dataset_dir.parent)
        payload["datasets"] = [{"id": "benchmark", "path": "datasets/benchmark.toml"}]
        comparison_entry: dict[str, object] = {
            "id": "benchmark-comparison",
            "matrix_dataset": "benchmark",
            "rhs_source": {"kind": "gaussian"},
        }
        if method_config_path is not None:
            comparison_entry["method"] = method_config_path.relative_to(path.parent).as_posix()
        payload["comparisons"] = [comparison_entry]
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return tracking_uri


def _write_method_config(path: Path) -> None:
    """Write a minimal comparison method override config."""
    path.write_text(
        '[general]\n\n[general.params]\nrtol = 1e-6\natol = 1e-14\nmax_iterations = 20\nstopping_criterion = "residual_norm"\nm_max = 5\n\n[[preconditioners]]\nname = "identity"\ntype = "identity"\n',
        encoding="utf-8",
    )


@pytest.mark.usefixtures("stub_input_gate")
@pytest.mark.integration
def test_comparison_logs_artifacts_to_mlflow_with_sqlite(tmp_path: Path) -> None:
    """Comparison workflow logs figures and diagnostics files under run artifacts."""
    output_root = tmp_path / "output"
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = tmp_path / "benchmark"
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, np.eye(2, dtype=np.float64))
    build_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 1},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )

    # Write a method config to test that config artifacts are logged under config/.
    method_config = tmp_path / "method.toml"
    _write_method_config(method_config)

    experiments_config = tmp_path / "experiments.toml"
    tracking_uri = _write_experiments_config(
        experiments_config,
        output_root,
        dataset_dir=dataset_dir,
        method_config_path=method_config,
    )

    # Ensure comparison experiment uses a deterministic local artifact root.
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="Comparisons",
        artifact_root=artifact_root,
    )
    legacy = client.create_run(experiment_id, tags={"phase": "comparison"})
    client.set_terminated(legacy.info.run_id, status="FINISHED")

    outcomes = run_comparison_batch(
        case_config_path=experiments_config,
        params=ComparisonParams(),
    )

    assert outcomes and outcomes[0].success is True

    # The session owns cross-method reports and has one direct method child.
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.phase = 'session_comparison'",
        max_results=1,
    )
    assert runs
    session_id = runs[0].info.run_id
    method_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{session_id}' and tags.phase = 'comparison_method'",
    )
    assert len(method_runs) == 1
    method_run = method_runs[0]
    assert not client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{method_run.info.run_id}'",
    )

    method_artifacts = client.list_artifacts(method_run.info.run_id, path="method_workload")
    method_names = {Path(item.path).name for item in method_artifacts}
    assert {"setup_events.parquet", "solve_attempts.parquet", "summary.toml"}.issubset(method_names)
    assert {
        "method_config.toml",
        "method_diagnostics.json",
        "residual_histories.json",
        "energy_histories.json",
    }.issubset(method_names)
    method_figures = client.list_artifacts(method_run.info.run_id, path="method_workload/figures")
    assert {Path(item.path).name for item in method_figures} == {
        "residual_history.png",
        "energy_history.png",
    }

    session_artifacts = client.list_artifacts(session_id, path="group_reports")
    assert any(
        Path(item.path).name.startswith("system_time_scatter_") for item in session_artifacts
    )
    legacy_runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.phase = 'comparison'"
    )
    assert [run.info.run_id for run in legacy_runs] == [legacy.info.run_id]

    session_count = len(
        client.search_runs(
            experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
        )
    )
    reused = run_comparison_batch(case_config_path=experiments_config, params=ComparisonParams())
    assert reused and reused[0].success is True
    assert (
        len(
            client.search_runs(
                experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == session_count
    )

    forced = run_comparison_batch(
        case_config_path=experiments_config, params=ComparisonParams(force=True)
    )
    assert forced and forced[0].success is True
    assert (
        len(
            client.search_runs(
                experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == session_count + 1
    )
