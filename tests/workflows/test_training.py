"""Tests for TrainingResult.stacked and TrainingResult.to_numpy() API.

Covers:
- stacked is None when no predictions captured
- to_numpy() returns None when no predictions captured
- stacked is a TensorDict when TensorDict predictions are present
- to_numpy() extracts predictions and targets correctly
- fast-dev-run integration: trainer.predict() produces stacked/to_numpy output
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from dlkit.common.results import TrainingResult
from tensordict import TensorDict

from neuralls.composition.assignments._training_artifacts import MlflowCoordinates
from neuralls.composition.assignments.assembler import AssignmentIdentity
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.assignments.training import (
    _finalize_training_run,
    _TrainingFinalizationContext,
)
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.artifact_access import ArtifactLease
from neuralls.platform.tracking.mlflow import MlflowRunConfig


class _CheckpointLeaseManager:
    """Minimal artifact lease manager for training checkpoint fallback tests."""

    def __init__(self, checkpoint_root: Path) -> None:
        self.checkpoint_root = checkpoint_root
        self.dir_calls: list[tuple[str, str]] = []

    @property
    def tracking_uri(self) -> str | None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        raise RuntimeError(f"Unexpected file lease for {run_id}:{artifact_path}")

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.dir_calls.append((run_id, artifact_path))
        return ArtifactLease(
            path=self.checkpoint_root,
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri="file:///artifacts",
            local_copy=False,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def predictions_as_dicts() -> list[TensorDict]:
    """Prediction batches in TensorDict predict_step format.

    Mirrors the structure dlkit predict_step returns:
    {"predictions": TensorDict({"output": tensor}), "targets": TensorDict({})}
    """
    return [
        TensorDict(
            {
                "predictions": TensorDict(
                    {"output": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3]
                ),
                "targets": TensorDict({}, batch_size=[3]),
            },
            batch_size=[3],
        ),
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([4.0, 5.0])}, batch_size=[2]),
                "targets": TensorDict({}, batch_size=[2]),
            },
            batch_size=[2],
        ),
    ]


@pytest.fixture
def predictions_as_tensors() -> list[torch.Tensor]:
    """Prediction batches as plain tensors (fallback format)."""
    return [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
    ]


@pytest.fixture
def training_result_no_predictions() -> TrainingResult:
    """TrainingResult with no captured predictions (None)."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=None,
    )


@pytest.fixture
def training_result_empty_predictions() -> TrainingResult:
    """TrainingResult with empty prediction list."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=[],
    )


@pytest.fixture
def training_result_dict_predictions(predictions_as_dicts: list[TensorDict]) -> TrainingResult:
    """TrainingResult with TensorDict-format predictions (no targets)."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_as_dicts,
    )


@pytest.fixture
def training_result_tensor_predictions(
    predictions_as_tensors: list[torch.Tensor],
) -> TrainingResult:
    """TrainingResult with plain tensor format predictions."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_as_tensors,
    )


@pytest.fixture
def predictions_with_targets() -> list[TensorDict]:
    """Prediction batches with matching targets in TensorDict format."""
    return [
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([1.0, 2.0])}, batch_size=[2]),
                "targets": TensorDict({"y": torch.tensor([1.1, 2.1])}, batch_size=[2]),
            },
            batch_size=[2],
        ),
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([3.0])}, batch_size=[1]),
                "targets": TensorDict({"y": torch.tensor([3.1])}, batch_size=[1]),
            },
            batch_size=[1],
        ),
    ]


@pytest.fixture
def training_result_with_targets(predictions_with_targets: list[TensorDict]) -> TrainingResult:
    """TrainingResult with both predictions and targets."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_with_targets,
    )


# ---------------------------------------------------------------------------
# stacked property tests
# ---------------------------------------------------------------------------


def test_stacked_none_when_no_predictions(
    training_result_no_predictions: TrainingResult,
) -> None:
    """result.stacked is None when predictions field is None."""
    assert training_result_no_predictions.stacked is None


def test_stacked_none_when_empty_predictions(
    training_result_empty_predictions: TrainingResult,
) -> None:
    """result.stacked is None when predictions is an empty list."""
    assert training_result_empty_predictions.stacked is None


def test_stacked_is_tensordict_for_tensordict_input(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """result.stacked is a TensorDict when TensorDict predictions are stored."""
    assert isinstance(training_result_dict_predictions.stacked, TensorDict)


# ---------------------------------------------------------------------------
# to_numpy() tests
# ---------------------------------------------------------------------------


def test_to_numpy_returns_none_when_no_predictions(
    training_result_no_predictions: TrainingResult,
) -> None:
    """to_numpy() returns None when no predictions were captured."""
    assert training_result_no_predictions.to_numpy() is None


def test_to_numpy_extracts_predictions_from_tensordict(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """to_numpy() extracts and concatenates output across TensorDict batches.

    Two batches of size 3 and 2 → flat ndarray of shape (5,).
    """
    result = training_result_dict_predictions.to_numpy()

    assert result is not None
    output = result.get("predictions", {}).get("output")
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape == (5,)
    np.testing.assert_allclose(output, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_to_numpy_extracts_targets_when_present(
    training_result_with_targets: TrainingResult,
) -> None:
    """to_numpy() extracts targets from TensorDict predictions."""
    result = training_result_with_targets.to_numpy()

    assert result is not None
    targets = result.get("targets", {})
    assert "y" in targets
    y = targets["y"]
    assert isinstance(y, np.ndarray)
    assert y.shape == (3,)
    np.testing.assert_allclose(y, [1.1, 2.1, 3.1], atol=1e-6)


def test_to_numpy_targets_empty_when_no_targets(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """to_numpy()['targets'] is empty when targets dicts are empty in all batches."""
    result = training_result_dict_predictions.to_numpy()
    assert result is not None
    targets = result.get("targets", {})
    assert targets == {}


def test_to_numpy_predictions_shape_matches_targets(
    training_result_with_targets: TrainingResult,
) -> None:
    """Predictions and targets arrays have matching shapes from to_numpy()."""
    result = training_result_with_targets.to_numpy()
    assert result is not None
    output = result["predictions"]["output"]
    y = result["targets"]["y"]
    assert output.shape == y.shape


# ---------------------------------------------------------------------------
# fast_dev_run integration: verify trainer.predict() structure
# ---------------------------------------------------------------------------


def test_fast_dev_run_predict_returns_list_of_dicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainer.predict() produces stacked TensorDict and to_numpy() output.

    This test documents the expected API so _log_training_evaluation() can rely on it.
    Runs a single fast_dev_run training step using DLKit programmatic settings.
    """
    from dlkit.config import TrainingJobConfig
    from dlkit.engine.tracking import uri_resolver
    from dlkit.infrastructure.config import (
        DataModuleSelector,
        DataSettings,
        ExperimentSettings,
        RunSettings,
        TrackingSettings,
        TrainingSettings,
    )
    from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry
    from dlkit.infrastructure.config.model_components import (
        MetricComponentSettings,
        ModelComponentSettings,
    )
    from dlkit.infrastructure.config.trainer_settings import TrainerSettings
    from dlkit.interfaces.api import execute

    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    n_samples, n_features, n_targets = 20, 4, 2
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_targets))

    settings = TrainingJobConfig(
        run=RunSettings(type="train", seed=42),
        experiment=ExperimentSettings(name="test_predict_structure"),
        tracking=TrackingSettings(backend="none"),
        data=DataSettings(
            name="FlexibleDataset",
            batch_size=4,
            num_workers=0,
            module=DataModuleSelector(name="ArrayDataModule"),
            features=(ValueEntry(name="x", value=X),),
            targets=(ValueEntry(name="y", value=Y, data_role=DataRole.TARGET),),
        ),
        training=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                default_root_dir=tmp_path,
            ),
            metrics=(
                MetricComponentSettings(
                    name="MeanSquaredError",
                    module_path="dlkit.domain.metrics",
                ),
            ),
        ),
        model=ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.nn",
            hidden_size=4,
            num_layers=1,
        ),
    )

    result = cast(TrainingResult, execute(settings))

    assert result.predictions is not None, "trainer.predict() should have captured predictions"
    assert isinstance(result.predictions, list)
    assert len(result.predictions) > 0

    stacked = result.stacked
    assert stacked is not None, "result.stacked should be a TensorDict after prediction"

    all_numpy = result.to_numpy()
    assert all_numpy is not None, "to_numpy() should return a dict after prediction"
    assert "predictions" in all_numpy, f"Expected 'predictions' key, got {list(all_numpy.keys())}"
    preds = all_numpy["predictions"]
    if isinstance(preds, dict):
        output = next(iter(preds.values()))
    else:
        output = preds
    assert isinstance(output, np.ndarray)
    assert output.ndim >= 1


def test_resolve_training_checkpoint_leases_mlflow_artifacts(
    tmp_path: Path,
) -> None:
    """MLflow fallback checkpoint resolution must not create workspace downloads."""
    from neuralls.composition.assignments.training import _resolve_training_checkpoint
    from neuralls.platform.config.models.workspace import AssignmentWorkspace

    workspace = AssignmentWorkspace(
        dataset_id="dataset",
        run_id="run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    training_result = SimpleNamespace(checkpoint_path=None, artifacts={})
    checkpoint_root = tmp_path / "mlflow-artifacts" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    resolved_checkpoint = checkpoint_root / "model.ckpt"
    resolved_checkpoint.write_text("checkpoint")
    artifact_leases = _CheckpointLeaseManager(checkpoint_root)

    with patch(
        "neuralls.composition.assignments._training_artifacts.get_latest_checkpoint",
        return_value=None,
    ):
        resolved = _resolve_training_checkpoint(
            training_result=training_result,
            workspace=workspace,
            run_id="run-1",
            artifact_leases=artifact_leases,
        )

    assert resolved == resolved_checkpoint
    assert artifact_leases.dir_calls == [("run-1", "checkpoints")]
    assert not (workspace.root_dir / "mlflow-downloads").exists()


def test_resolve_tracking_backend_returns_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend comes from tracking.toml, and an explicit URI passes the guard."""
    import neuralls.composition.assignments.training as training_module
    from neuralls.platform.tracking.mlflow import MlflowRuntimeEnvironment

    monkeypatch.setattr(
        training_module,
        "load_tracking_config",
        lambda: SimpleNamespace(backend="mlflow"),
    )
    runtime_environment = MlflowRuntimeEnvironment(
        env={"MLFLOW_TRACKING_URI": "http://tracking.test:5000"},
        tracking_uri="http://tracking.test:5000",
        artifact_uri=None,
        is_explicit=True,
    )

    assert training_module._resolve_tracking_backend(runtime_environment) == "mlflow"


def test_resolve_tracking_backend_rejects_implicit_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuses to silently train against the local sqlite fallback URI."""
    import neuralls.composition.assignments.training as training_module
    from neuralls.platform.tracking.mlflow import MlflowRuntimeEnvironment

    monkeypatch.setattr(
        training_module,
        "load_tracking_config",
        lambda: SimpleNamespace(backend="mlflow"),
    )
    runtime_environment = MlflowRuntimeEnvironment(
        env={"MLFLOW_TRACKING_URI": "sqlite:////fallback/mlruns/mlflow.db"},
        tracking_uri="sqlite:////fallback/mlruns/mlflow.db",
        artifact_uri=None,
        is_explicit=False,
    )

    with pytest.raises(RuntimeError, match="no explicit tracking URI"):
        training_module._resolve_tracking_backend(runtime_environment)


def test_prepare_training_settings_builds_explicit_mlflow_run_config(tmp_path: Path) -> None:
    """Registry-backed preparation builds explicit experiment/run names and structured tags.

    Targets ``prepare_training_settings()`` directly rather than ``train_model()``
    (removed — both batch orchestrators now call ``prepare_training_settings()``/
    ``finalize_prepared_training()`` directly instead of the old single-shot
    wrapper), so only the settings-building half needs exercising here; no
    execute()/finalize mocks are needed since this function never calls either.
    """
    from neuralls.composition.assignments.training import (
        cleanup_prepared_training,
        prepare_training_settings,
    )
    from neuralls.platform.config.models.workspace import AssignmentWorkspace

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")

    workspace = AssignmentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)

    experiment = SimpleNamespace(
        settings=SimpleNamespace(data=None),
        workspace=workspace,
        spec=SimpleNamespace(
            assignment_id="exp-1",
            assignment_display_name="Experiment One",
            dataset_id="dataset-1",
            dataset_display_name="Dataset One",
            job_id="job-1",
            job_display_name="Job One",
        ),
    )

    with (
        patch(
            "neuralls.composition.assignments.training.load_assignment", return_value=experiment
        ) as mock_load,
        patch(
            "neuralls.composition.assignments.training._load_and_prepare_data",
            return_value=(
                SimpleNamespace(rhs_source=SimpleNamespace(path=Path("rhs.npy"))),
                [],
                [],
            ),
        ),
        patch(
            "neuralls.composition.assignments.training._configure_training_pipeline",
            return_value=(experiment.settings, workspace),
        ),
        patch(
            "neuralls.composition.assignments.training.patch_model",
            side_effect=lambda s, _: s,
        ),
    ):
        prepared = prepare_training_settings(
            config_path=str(config_path),
            data_config_path=str(data_config_path),
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(
                assignment_id="exp-1",
                assignment_display_name="Experiment One",
                dataset_registry_id="dataset-1",
                dataset_display_name="Dataset One",
                job_registry_id="job-1",
                job_display_name="Job One",
            ),
            mlflow_experiment_name="CustomTrain",
        )
    cleanup_prepared_training(prepared)

    load_kwargs = mock_load.call_args.kwargs
    assert load_kwargs["job_config_path"] == config_path
    assert load_kwargs["data_config_path"] == data_config_path

    run_config = prepared.run_config
    assert run_config.experiment_name == "CustomTrain"
    assert re.match(
        r"^Experiment One \| [A-Z][a-z]{2} \d{2} [A-Z][a-z]{2} \d{4} - \d{2}:\d{2}:\d{2}$",
        run_config.run_name,
    )
    assert run_config.tags == {
        "phase": "training",
        "assignment_id": "exp-1",
        "dataset_id": "dataset-1",
        "job_id": "job-1",
        "assignment_display_name": "Experiment One",
    }


def test_prepare_training_settings_falls_back_to_dataset_display_name_without_structured_tags(
    tmp_path: Path,
) -> None:
    """Legacy callers use the config-model default experiment name and no tags."""
    from neuralls.composition.assignments.training import (
        cleanup_prepared_training,
        prepare_training_settings,
    )
    from neuralls.platform.config.models.workspace import AssignmentWorkspace

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")

    workspace = AssignmentWorkspace(
        dataset_id="dataset-legacy",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)

    experiment = SimpleNamespace(
        settings=SimpleNamespace(data=None),
        workspace=workspace,
        spec=SimpleNamespace(
            assignment_id="legacy-exp",
            assignment_display_name="Legacy Experiment",
            dataset_id="dataset-legacy",
            dataset_display_name="Dataset Display",
            job_id=None,
            job_display_name=None,
        ),
    )

    with (
        patch("neuralls.composition.assignments.training.load_assignment", return_value=experiment),
        patch(
            "neuralls.composition.assignments.training._load_and_prepare_data",
            return_value=(
                SimpleNamespace(rhs_source=SimpleNamespace(path=Path("rhs.npy"))),
                [],
                [],
            ),
        ),
        patch(
            "neuralls.composition.assignments.training._configure_training_pipeline",
            return_value=(experiment.settings, workspace),
        ),
        patch(
            "neuralls.composition.assignments.training.patch_model",
            side_effect=lambda s, _: s,
        ),
    ):
        prepared = prepare_training_settings(
            config_path=config_path,
            data_config_path=data_config_path,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(
                assignment_id="legacy-exp",
                assignment_display_name="Legacy Experiment",
                dataset_registry_id="dataset-legacy",
                dataset_display_name="Dataset Display",
            ),
        )
    cleanup_prepared_training(prepared)

    run_config = prepared.run_config
    assert run_config.experiment_name == ExperimentNamesConfig().training
    assert re.match(
        r"^Legacy Experiment \| [A-Z][a-z]{2} \d{2} [A-Z][a-z]{2} \d{4} - \d{2}:\d{2}:\d{2}$",
        run_config.run_name,
    )
    assert run_config.tags == {}


def test_prepare_training_settings_max_epochs_override_keeps_original_settings_immutable(
    tmp_path: Path,
) -> None:
    """`max_epochs` override patches a fresh settings object, leaving the original untouched."""
    from dlkit.infrastructure.config import (
        DataModuleSelector,
        DataSettings,
        ExperimentSettings,
        RunSettings,
        TrackingSettings,
        TrainingSettings,
    )
    from dlkit.infrastructure.config.job_config import TrainingJobConfig
    from dlkit.infrastructure.config.model_components import ModelComponentSettings
    from dlkit.infrastructure.config.trainer_settings import TrainerSettings

    from neuralls.composition.assignments.training import (
        cleanup_prepared_training,
        prepare_training_settings,
    )
    from neuralls.platform.config.models.workspace import AssignmentWorkspace

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")
    trainer_root = tmp_path / "trainer-root"
    trainer_root.mkdir()

    base_settings = TrainingJobConfig(
        run=RunSettings(type="train", seed=42),
        experiment=ExperimentSettings(name="exp-1"),
        tracking=TrackingSettings(backend="none"),
        model=ModelComponentSettings(name="LinearModel"),
        data=DataSettings(
            name="FlexibleDataset",
            module=DataModuleSelector(name="ArrayDataModule"),
        ),
        training=TrainingSettings(
            trainer=TrainerSettings(max_epochs=1, default_root_dir=trainer_root)
        ),
    )

    workspace = AssignmentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)

    experiment = SimpleNamespace(
        settings=base_settings,
        workspace=workspace,
        spec=SimpleNamespace(
            assignment_id="exp-1",
            assignment_display_name="Experiment One",
            dataset_id="dataset-1",
            dataset_display_name="Dataset One",
            job_id=None,
            job_display_name=None,
        ),
    )

    with (
        patch("neuralls.composition.assignments.training.load_assignment", return_value=experiment),
        patch(
            "neuralls.composition.assignments.training._load_and_prepare_data",
            return_value=(
                SimpleNamespace(rhs_source=SimpleNamespace(path=Path("rhs.npy"))),
                [],
                [],
            ),
        ),
        patch(
            "neuralls.composition.assignments.training._configure_training_pipeline",
            return_value=(base_settings, workspace),
        ),
    ):
        prepared = prepare_training_settings(
            config_path=config_path,
            data_config_path=data_config_path,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(
                assignment_id="exp-1",
                assignment_display_name="Experiment One",
                dataset_registry_id="dataset-1",
                dataset_display_name="Dataset One",
            ),
            max_epochs=9,
        )
    cleanup_prepared_training(prepared)

    assert prepared.workflow_settings.training is not None
    assert prepared.workflow_settings.training.trainer is not None
    assert prepared.workflow_settings.training.trainer.max_epochs == 9
    assert base_settings.training is not None
    assert base_settings.training.trainer is not None
    assert base_settings.training.trainer.max_epochs == 1


def test_execute_result_unwraps_optimization_result() -> None:
    """Optimization results are normalized to their nested training result."""
    from dlkit.common.results import OptimizationResult, TrialRecord

    from neuralls.composition.assignments.training import _unwrap_execution_result

    training_result = TrainingResult(
        model_state=None,
        metrics={"val/loss": 0.1},
        artifacts={},
        duration_seconds=1.0,
    )
    optimization_result = OptimizationResult(
        best_trial=TrialRecord(number=1, value=0.1, params={}, state="COMPLETE"),
        training_result=training_result,
        study_summary={},
        duration_seconds=2.0,
    )

    assert _unwrap_execution_result(training_result) is training_result
    assert _unwrap_execution_result(optimization_result) is training_result


# ---------------------------------------------------------------------------
# _finalize_training_run tests
#
# Narrow tests against _finalize_training_run directly (not through the full
# train_model() mocking harness) — the extraction's whole point is that
# reaching a Step 6-8 failure path no longer requires mocking the upstream
# data-loading/execute() machinery.
# ---------------------------------------------------------------------------


@pytest.fixture
def finalize_workspace(tmp_path: Path) -> AssignmentWorkspace:
    """Assignment workspace for _finalize_training_run tests."""
    workspace = AssignmentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    return workspace


@pytest.fixture
def finalize_assignment() -> SimpleNamespace:
    """Minimal assignment collaborator exposing the .spec fields _finalize_training_run reads."""
    return SimpleNamespace(
        spec=SimpleNamespace(
            assignment_id="exp-1",
            dataset_id="dataset-1",
            job_id="job-1",
            job_display_name="Job One",
        )
    )


@pytest.fixture
def finalize_run_config(tmp_path: Path) -> MlflowRunConfig:
    """MlflowRunConfig for _finalize_training_run tests."""
    return MlflowRunConfig(
        experiment_name="CustomTrain",
        run_name="Experiment One | run",
        tags={},
        paths=MlflowPaths(tracking_uri="sqlite:///unused.db", artifact_uri=None),
        workspace_root=tmp_path / "workspace",
    )


@pytest.fixture
def finalize_config_path(tmp_path: Path) -> Path:
    """Job config TOML path for _finalize_training_run tests."""
    config_path = tmp_path / "model.toml"
    config_path.write_text("")
    return config_path


@pytest.fixture
def finalize_context(
    finalize_workspace: AssignmentWorkspace,
    finalize_assignment: SimpleNamespace,
    finalize_run_config: MlflowRunConfig,
    finalize_config_path: Path,
) -> _TrainingFinalizationContext:
    """Bundled context for _finalize_training_run tests, with a resolvable run_id."""
    return _TrainingFinalizationContext(
        training_result=SimpleNamespace(run_id="run-123", metrics={}),
        run_config=finalize_run_config,
        workspace=finalize_workspace,
        assignment=finalize_assignment,
        assignment_id="exp-1",
        resolved_assignment_display_name="Experiment One",
        dataset_id="dataset-1",
        resolved_dataset_display_name="Dataset One",
        workflow_settings=SimpleNamespace(data=None),
        contract=default_training_dataset_contract(),
        config_path=finalize_config_path,
        resolved_data_config_path=None,
        fallback_tracking_uri="sqlite:///fallback.db",
    )


def test_finalize_training_run_happy_path_returns_run_coords(
    finalize_context: _TrainingFinalizationContext,
) -> None:
    """All Step 6-8 collaborators succeed: returns (run_id, tracking_uri); no failure marking."""
    resolved_coords = MlflowCoordinates("sqlite:///resolved.db", "mlflow-exp-1", "run-123")
    checkpoint_path = finalize_context.workspace.checkpoint_dir / "model.ckpt"

    with (
        patch(
            "neuralls.composition.assignments.training._resolve_mlflow_run_ids",
            return_value=resolved_coords,
        ),
        patch(
            "neuralls.composition.assignments.training._resolve_training_checkpoint",
            return_value=checkpoint_path,
        ),
        patch("neuralls.composition.assignments.training._log_training_context"),
        patch("neuralls.composition.assignments.training.ensure_checkpoint_artifact"),
        patch("neuralls.composition.assignments.training._stage_training_artifacts"),
        patch("neuralls.composition.assignments.training._log_training_evaluation"),
        patch("neuralls.composition.assignments.training.log_artifacts_to_mlflow"),
        patch("neuralls.composition.assignments.training.log_extra_feature_names_tag"),
        patch("neuralls.composition.assignments.training.mark_run_failed") as mock_mark_failed,
    ):
        result = _finalize_training_run(finalize_context)

    assert result == MlflowCoordinates("sqlite:///resolved.db", "mlflow-exp-1", "run-123")
    mock_mark_failed.assert_not_called()


def test_finalize_training_run_marks_run_failed_and_reraises_on_durability_failure(
    finalize_context: _TrainingFinalizationContext,
) -> None:
    """A durability step (ensure_checkpoint_artifact) raising marks the run FAILED and re-raises.

    The exact same exception instance must propagate — mark_run_failed must not
    swallow or wrap it.
    """
    resolved_coords = MlflowCoordinates("sqlite:///resolved.db", "mlflow-exp-1", "run-123")
    checkpoint_path = finalize_context.workspace.checkpoint_dir / "model.ckpt"
    original_exc = RuntimeError("checkpoint upload failed")

    with (
        patch(
            "neuralls.composition.assignments.training._resolve_mlflow_run_ids",
            return_value=resolved_coords,
        ),
        patch(
            "neuralls.composition.assignments.training._resolve_training_checkpoint",
            return_value=checkpoint_path,
        ),
        patch("neuralls.composition.assignments.training._log_training_context"),
        patch(
            "neuralls.composition.assignments.training.ensure_checkpoint_artifact",
            side_effect=original_exc,
        ),
        patch("neuralls.composition.assignments.training._stage_training_artifacts"),
        patch("neuralls.composition.assignments.training._log_training_evaluation"),
        patch("neuralls.composition.assignments.training.log_artifacts_to_mlflow"),
        patch("neuralls.composition.assignments.training.log_extra_feature_names_tag"),
        patch("neuralls.composition.assignments.training.mark_run_failed") as mock_mark_failed,
        pytest.raises(RuntimeError, match="checkpoint upload failed") as exc_info,
    ):
        _finalize_training_run(finalize_context)

    assert exc_info.value is original_exc
    mock_mark_failed.assert_called_once_with(run_id="run-123", tracking_uri="sqlite:///resolved.db")


def test_finalize_training_run_raises_without_marking_when_no_run_established(
    finalize_context: _TrainingFinalizationContext,
) -> None:
    """No MLflow run at all (no coords, fallback creation also fails): raises RuntimeError.

    mark_run_failed must not be called since no run_id was ever resolved — there
    is nothing to mark.
    """
    context = replace(
        finalize_context,
        training_result=SimpleNamespace(metrics={}),  # no run_id/mlflow_run_id attrs
    )
    checkpoint_path = context.workspace.checkpoint_dir / "model.ckpt"

    with (
        patch(
            "neuralls.composition.assignments.training._resolve_mlflow_run_ids",
            return_value=None,
        ),
        patch(
            "neuralls.composition.assignments.training._resolve_finalization_checkpoint",
            return_value=checkpoint_path,
        ),
        patch(
            "neuralls.composition.assignments.training.create_fallback_training_run",
            side_effect=RuntimeError("no mlflow server reachable"),
        ),
        patch("neuralls.composition.assignments.training.mark_run_failed") as mock_mark_failed,
        pytest.raises(RuntimeError, match="exp-1"),
    ):
        _finalize_training_run(context)

    mock_mark_failed.assert_not_called()
