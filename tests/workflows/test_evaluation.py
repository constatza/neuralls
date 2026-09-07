"""Tests for eval-only workflow assembly."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dlkit.common import ChildSuccess
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig

from neuralls.composition.assignments.evaluation import (
    EvaluationAssignmentContext,
    EvaluationBatchResult,
    EvaluationConfigPaths,
    EvaluationRunResult,
    PreparedEvaluation,
    _as_inference_job,
    _finalize_eval_child,
    _materialize_inference_settings,
    _with_eval_runtime_dataset,
    eval_batch,
    to_eval_run_spec,
    write_eval_metric_report,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.config.loaders import load_case_config
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.tracking.evaluation_artifacts import (
    TrainingConfigArtifacts,
    TrainingEvaluationArtifacts,
)

SPLIT_ARTIFACT_PATH = "splits/split.json"


def _write_split(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
                "predict": [],
            }
        ),
        encoding="utf-8",
    )


def test_eval_materialization_injects_split_file_and_array_datamodule(
    tmp_path: Path, minimal_training_job: TrainingJobConfig
) -> None:
    job = minimal_training_job
    split_file = tmp_path / "split.json"
    checkpoint = tmp_path / "model.ckpt"
    rhs = tmp_path / "rhs.npy"
    solution = tmp_path / "solutions.npy"
    matrix = tmp_path / "matrix.npy"
    _write_split(split_file)
    checkpoint.write_bytes(b"checkpoint")
    np.save(rhs, np.zeros((4, 3)))
    np.save(solution, np.ones((4, 3)))
    np.save(matrix, np.eye(3))
    contract = default_training_dataset_contract()
    features = [
        entry_from_path(
            rhs,
            name=contract.primary_input_name,
            model_input=True,
            role="feature",
        ),
        entry_from_path(
            matrix,
            name=contract.matrix_input_name,
            model_input=False,
            role="feature",
        ),
    ]
    targets = [
        entry_from_path(
            solution,
            name=contract.target_name,
            model_input=False,
            role="target",
        )
    ]

    eval_job = _with_eval_runtime_dataset(
        job,
        split_file=split_file,
        checkpoint_path=checkpoint,
        features=features,
        targets=targets,
        dataset_format="npy",
    )
    inference_job = _as_inference_job(eval_job)

    assert isinstance(inference_job, InferenceJobConfig)
    assert inference_job.run.type == "predict"
    assert inference_job.model.checkpoint == str(checkpoint)
    assert inference_job.data is not None
    assert inference_job.data.splits.filepath == split_file
    assert inference_job.data.module.name == "ArrayDataModule"
    assert inference_job.data.module.module_path == "dlkit.engine.adapters.lightning.datamodules"
    assert [entry.name for entry in inference_job.data.features] == ["x", "matrix"]
    assert [entry.name for entry in inference_job.data.targets] == ["y"]


def test_write_eval_metric_report_accepts_plain_and_prefixed_metric(tmp_path: Path) -> None:
    batch = EvaluationBatchResult(
        results=[
            EvaluationRunResult(
                label="1",
                assignment_id="a1",
                assignment_display_name="A1",
                training_run_id="train-1",
                evaluation_run_id="eval-1",
                metrics={"mae": 0.2, "eval/rmse": 0.4},
                split_file=tmp_path / "split.json",
                split_artifact_path=SPLIT_ARTIFACT_PATH,
                figures_dir=tmp_path / "figures",
            )
        ],
        label_map={},
        output_dir=tmp_path,
        tracking_uri=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        parent_run_id="parent-run-1",
    )

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        plotted = write_eval_metric_report(batch, metric="rmse")

    assert plotted is True
    logged_paths = {Path(call.args[1]).name for call in client.log_artifact.call_args_list}
    assert logged_paths == {"batch_metric_rmse.png", "batch_eval_labels.json"}
    assert all(call.args[0] == "parent-run-1" for call in client.log_artifact.call_args_list)


def test_materialize_inference_settings_uses_staged_config_paths(
    monkeypatch,
    tmp_path: Path,
    minimal_training_job: TrainingJobConfig,
) -> None:
    job = minimal_training_job
    staged_job = tmp_path / "staged-job.toml"
    staged_data = tmp_path / "staged-data.toml"
    current_job = tmp_path / "current-job.toml"
    current_data = tmp_path / "current-data.toml"
    split_file = tmp_path / "split.json"
    checkpoint = tmp_path / "model.ckpt"
    _write_split(split_file)
    checkpoint.write_bytes(b"checkpoint")
    for path in (staged_job, staged_data, current_job, current_data):
        path.write_text("", encoding="utf-8")
    load_assignment = MagicMock(
        return_value=SimpleNamespace(workspace=SimpleNamespace(data_dir=tmp_path))
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation.load_assignment",
        load_assignment,
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation.load_experiment_job",
        lambda path, settings: job,
    )
    np.save(tmp_path / "rhs.npy", np.zeros((4, 3)))
    np.save(tmp_path / "solutions.npy", np.ones((4, 3)))
    contract = default_training_dataset_contract()
    features = [
        entry_from_path(
            tmp_path / "rhs.npy",
            name=contract.primary_input_name,
            model_input=True,
            role="feature",
        )
    ]
    targets = [
        entry_from_path(
            tmp_path / "solutions.npy",
            name=contract.target_name,
            model_input=False,
            role="target",
        )
    ]
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation._load_and_prepare_data",
        lambda settings, workspace, contract: (
            SimpleNamespace(rhs_source=SimpleNamespace(path=Path("rhs.npy"))),
            features,
            targets,
        ),
    )
    context = EvaluationAssignmentContext(
        client=MagicMock(),
        training_run_id="train-1",
        artifacts=TrainingEvaluationArtifacts(
            checkpoint_path=checkpoint,
            split_file=split_file,
            split_artifact_path=SPLIT_ARTIFACT_PATH,
            config_artifacts=TrainingConfigArtifacts(config_dir=None),
        ),
        config_paths=EvaluationConfigPaths(
            job_config_path=current_job,
            data_config_path=current_data,
            staged_job_config_path=staged_job,
            staged_data_config_path=staged_data,
        ),
        dataset_display_name="Dataset",
        job_display_name="Job",
    )

    _materialize_inference_settings(
        cfg=CaseConfig(),
        settings=MagicMock(),
        assignment=AssignmentEntry(id="a1", dataset="d1", job="j1"),
        output_root=tmp_path,
        case_config_path=tmp_path / "case.toml",
        context=context,
    )

    assert load_assignment.call_args.kwargs["job_config_path"] == staged_job
    assert load_assignment.call_args.kwargs["data_config_path"] == staged_data


@pytest.fixture
def two_assignment_case_config(tmp_path: Path) -> Path:
    """Case config TOML with two eval assignments, for partial-failure tests."""
    path = tmp_path / "case.toml"
    path.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"',
                "",
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[assignments]]",
                'id = "assign-ok"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
                "[[assignments]]",
                'id = "assign-fail"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
            ]
        )
    )
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "ffnn.toml").write_text("")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "test-solutions.toml").write_text('id = "test-solutions"\n')
    return path


def _fake_prepared_evaluation(tmp_path: Path, assignment_id: str) -> PreparedEvaluation:
    """Minimal ``PreparedEvaluation`` stand-in for these integration tests.

    Mirrors ``test_runner.py``'s ``_fake_prepared_training``: a real
    ``PreparedEvaluation`` (needed so ``_finalize_eval_child`` can read its
    fields) with a duck-typed ``AssignmentEntry`` and a mocked MLflow client —
    the real settings-resolution pipeline is skipped by mocking
    ``prepare_evaluation_settings``/``run_multirun_spec``.
    """
    return PreparedEvaluation(
        inference_settings=MagicMock(),
        assignment=AssignmentEntry(id=assignment_id, dataset="d1", job="j1"),
        resolved_assignment_display_name=assignment_id,
        training_run_id="train-1",
        client=MagicMock(),
        split_file=tmp_path / "split.json",
        split_artifact_path=SPLIT_ARTIFACT_PATH,
        output_root=tmp_path,
        label="1",
        run_name=assignment_id,
    )


def test_eval_batch_continues_after_one_assignment_fails(
    two_assignment_case_config: Path,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """One failing assignment must not discard the rest of the batch's results.

    Mirrors run_assignment_sweep's resilience (training_batch.py): a per-assignment
    failure is logged and skipped, not allowed to abort the whole batch and
    lose already-completed reporting.
    """
    cfg = load_case_config(two_assignment_case_config, neuralls_settings)
    prepared_ok = _fake_prepared_evaluation(tmp_path, "assign-ok")
    fake_result = SimpleNamespace(mlflow_run_id="eval-1", metrics={"mae": 0.1}, figures={})
    sweep_result = SimpleNamespace(
        parent_run_id="parent-run-1",
        tracking_uri=cfg.mlflow.tracking_uri,
        children=(
            ChildSuccess(
                child_id="assign-ok", label="assign-ok", run_id="eval-1", result=fake_result
            ),
        ),
    )

    def _fake_prepare(*, assignment, **kwargs):
        if assignment.id == "assign-fail":
            raise RuntimeError("checkpoint missing")
        return prepared_ok

    with (
        patch(
            "neuralls.composition.assignments.evaluation.prepare_evaluation_settings",
            side_effect=_fake_prepare,
        ) as mock_prepare,
        patch(
            "neuralls.composition.assignments.evaluation.run_multirun_spec",
            return_value=sweep_result,
        ) as mock_sweep,
        patch("neuralls.composition.assignments.evaluation.finalize_session_parent_run"),
    ):
        batch = eval_batch(
            cfg=cfg,
            configs_dir=two_assignment_case_config.parent,
            settings=neuralls_settings,
            case_config_path=two_assignment_case_config,
        )

    assert mock_prepare.call_count == 2
    assert mock_sweep.called
    assert [result.assignment_id for result in batch.results] == ["assign-ok"]

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        plotted = write_eval_metric_report(batch, metric="mae")

    assert plotted is True
    logged_paths = {Path(call.args[1]).name for call in client.log_artifact.call_args_list}
    assert logged_paths == {"batch_metric_mae.png", "batch_eval_labels.json"}


def test_eval_batch_falls_back_to_own_parent_run_when_every_assignment_fails(
    two_assignment_case_config: Path,
    neuralls_settings,
) -> None:
    """No RunSpecs to dispatch still needs one MLflow run for the summary report.

    dlkit rejects an empty sweep, so when every assignment fails in phase 1,
    eval_batch must fall back to creating its own session-parent run rather
    than calling run_multirun_spec at all.
    """
    cfg = load_case_config(two_assignment_case_config, neuralls_settings)

    with (
        patch(
            "neuralls.composition.assignments.evaluation.prepare_evaluation_settings",
            side_effect=RuntimeError("checkpoint missing"),
        ),
        patch("neuralls.composition.assignments.evaluation.run_multirun_spec") as mock_sweep,
        patch(
            "neuralls.composition.assignments.evaluation.create_session_parent_run",
            return_value="fallback-parent-run",
        ) as mock_create,
        patch(
            "neuralls.composition.assignments.evaluation.finalize_session_parent_run"
        ) as mock_finalize,
    ):
        batch = eval_batch(
            cfg=cfg,
            configs_dir=two_assignment_case_config.parent,
            settings=neuralls_settings,
            case_config_path=two_assignment_case_config,
        )

    assert not mock_sweep.called
    assert mock_create.called
    assert batch.results == []
    assert batch.parent_run_id == "fallback-parent-run"
    assert mock_finalize.call_args.kwargs["status"] == "FAILED"


def test_to_eval_run_spec_builds_run_spec_from_prepared_evaluation(tmp_path: Path) -> None:
    """to_eval_run_spec() carries each child's own resolved settings into the sweep."""
    settings_sentinel = object()
    prepared = PreparedEvaluation(
        inference_settings=settings_sentinel,
        assignment=AssignmentEntry(id="assign-ok", dataset="d1", job="j1"),
        resolved_assignment_display_name="Assign OK",
        training_run_id="train-1",
        client=MagicMock(),
        split_file=tmp_path / "split.json",
        split_artifact_path=SPLIT_ARTIFACT_PATH,
        output_root=tmp_path,
        label="1",
        run_name="Assign OK",
    )

    run_spec = to_eval_run_spec(prepared)

    assert run_spec.id == "assign-ok"
    assert run_spec.label == "Assign OK"
    assert run_spec.settings is settings_sentinel
    assert run_spec.run_name == "Assign OK"


def test_finalize_eval_child_saves_outputs_and_tags_run(tmp_path: Path) -> None:
    """_finalize_eval_child saves metrics/figures and tags the child's MLflow run."""
    mock_client = MagicMock()
    prepared = _fake_prepared_evaluation(tmp_path, "assign-ok")
    prepared = replace(prepared, client=mock_client)
    fake_result = SimpleNamespace(mlflow_run_id="eval-1", metrics={"mae": 0.1}, figures={})

    result = _finalize_eval_child(prepared, fake_result)

    assert result.assignment_id == "assign-ok"
    assert result.evaluation_run_id == "eval-1"
    assert result.metrics == {"mae": 0.1}
    metrics_file = tmp_path / "eval" / "assign-ok" / "metrics" / "evaluation_metrics.json"
    assert metrics_file.exists()
    assert mock_client.set_tag.called
