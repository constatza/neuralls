"""Characterization tests for training MLflow-run/checkpoint resolution helpers.

Pins the observable behavior of the `_training_artifacts` helpers that had no
direct coverage before the training-orchestration refactor:

- `_build_training_run_config` — structured vs. ad-hoc run naming/tags.
- `_finalize_existing_mlflow_run` / `_finalize_fallback_mlflow_run` — which
  MLflow coordinates come back out of each finalization branch.

These assert *values*, not signatures, so they stay meaningful across the
parameter-object/DTO refactor of the same call chain.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments._training_artifacts import (
    MlflowCoordinates,
    _build_training_run_config,
)
from neuralls.composition.assignments.assembler import AssignmentIdentity
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.assignments.training import (
    _finalize_existing_mlflow_run,
    _finalize_fallback_mlflow_run,
    _TrainingFinalizationContext,
)
from neuralls.composition.tracking.run_specs import training_run_name
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.tracking.mlflow import MlflowPaths, MlflowRunConfig
from neuralls.platform.tracking.run_names import run_name_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime_mlflow_env() -> dict[str, str]:
    """Active MLflow environment mapping for run-config construction."""
    return {
        "MLFLOW_TRACKING_URI": "sqlite:///runtime.db",
        "MLFLOW_ARTIFACT_URI": "file:///artifacts",
    }


@pytest.fixture
def workspace_root() -> Path:
    """Workspace root directory for run-config construction."""
    return Path("workspace")


@pytest.fixture
def registry_identity() -> AssignmentIdentity:
    """Fully registry-backed identity (assignment + dataset + job ids all set)."""
    return AssignmentIdentity(
        assignment_id="exp-1",
        dataset_registry_id="dataset-1",
        job_registry_id="job-1",
    )


@pytest.fixture
def adhoc_identity() -> AssignmentIdentity:
    """Ad-hoc identity with no registry ids — the non-structured naming path."""
    return AssignmentIdentity()


@pytest.fixture
def finalization_workspace() -> AssignmentWorkspace:
    """Workspace layout for the finalization branches; nothing exists on disk."""
    return AssignmentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=Path("ws"),
        data_dir=Path("ws/data"),
    )


@pytest.fixture
def finalization_context(
    finalization_workspace: AssignmentWorkspace,
    training_presentation: TrainingPresentation,
) -> _TrainingFinalizationContext:
    """Minimal finalization context for the two `_finalize_*_mlflow_run` branches."""
    config_path = Path("job.toml")
    return _TrainingFinalizationContext(
        tracking=MagicMock(),
        training_result=SimpleNamespace(metrics={}),
        run_config=MlflowRunConfig(
            experiment_name="Training",
            name=training_run_name(training_presentation, include_timestamp=False),
            tags={"phase": "training"},
            paths=MlflowPaths(tracking_uri="sqlite:///unused.db", artifact_uri=None),
            workspace_root=finalization_workspace.root_dir,
        ),
        workspace=finalization_workspace,
        assignment=SimpleNamespace(
            spec=SimpleNamespace(
                assignment_id="exp-1",
                dataset_id="dataset-1",
                job_id="job-1",
            )
        ),
        assignment_id="exp-1",
        presentation=training_presentation,
        dataset_id="dataset-1",
        workflow_settings=SimpleNamespace(data=None),
        contract=default_training_dataset_contract(),
        config_path=config_path,
        resolved_data_config_path=None,
        fallback_tracking_uri="sqlite:///fallback.db",
    )


TRAINED_TEXT = "POD | CG Errors (k=20, N=100)"


# ---------------------------------------------------------------------------
# _build_training_run_config
# ---------------------------------------------------------------------------


def test_build_training_run_config_uses_structured_tags_when_all_registry_ids_present(
    registry_identity: AssignmentIdentity,
    training_presentation: TrainingPresentation,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """A fully registry-backed identity produces the structured training run spec."""
    config = _build_training_run_config(
        identity=registry_identity,
        presentation=training_presentation,
        experiment_name="CustomTraining",
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=False,
    )

    assert config.experiment_name == "CustomTraining"
    assert run_name_text(config.name) == TRAINED_TEXT
    assert config.tags["assignment_id"] == "exp-1"
    assert config.tags["dataset_id"] == "dataset-1"
    assert config.tags["job_id"] == "job-1"
    assert config.paths.tracking_uri == "sqlite:///runtime.db"
    assert config.paths.artifact_uri == "file:///artifacts"
    assert config.workspace_root == workspace_root


def test_build_training_run_config_appends_timestamp_when_requested(
    registry_identity: AssignmentIdentity,
    training_presentation: TrainingPresentation,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """include_timestamp=True suffixes the display name with ' | <timestamp>'."""
    config = _build_training_run_config(
        identity=registry_identity,
        presentation=training_presentation,
        experiment_name="CustomTraining",
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=True,
    )

    text = run_name_text(config.name)
    title, _, stamp = text.rpartition(" | ")
    assert title == TRAINED_TEXT
    assert stamp != ""


def test_build_training_run_config_falls_back_to_untagged_spec_without_registry_ids(
    adhoc_identity: AssignmentIdentity,
    training_presentation: TrainingPresentation,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """Missing registry ids drop to a bare, untagged run config named by what it trains."""
    config = _build_training_run_config(
        identity=adhoc_identity,
        presentation=training_presentation,
        experiment_name="CustomTraining",
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=False,
    )

    assert run_name_text(config.name) == TRAINED_TEXT
    assert dict(config.tags) == {}


# ---------------------------------------------------------------------------
# _finalize_existing_mlflow_run / _finalize_fallback_mlflow_run
# ---------------------------------------------------------------------------


def test_finalize_existing_mlflow_run_returns_the_coordinates_it_was_given(
    finalization_context: _TrainingFinalizationContext,
) -> None:
    """The existing-run branch makes the run durable and echoes its own coordinates back."""
    checkpoint_path = Path("ws/checkpoints/model.ckpt")

    with (
        patch("neuralls.composition.assignments.training._log_training_context") as log_context,
        patch("neuralls.composition.assignments.training.log_extra_feature_names_tag"),
        patch(
            "neuralls.composition.assignments.training.ensure_checkpoint_artifact"
        ) as ensure_checkpoint,
        patch("neuralls.composition.assignments.training._stage_training_artifacts"),
        patch("neuralls.composition.assignments.training._log_training_evaluation"),
        patch("neuralls.composition.assignments.training.log_artifacts_to_mlflow") as log_artifacts,
    ):
        coords = _finalize_existing_mlflow_run(
            context=finalization_context,
            mlflow_coords=MlflowCoordinates("sqlite:///resolved.db", "mlflow-exp-1", "run-123"),
            checkpoint_path=checkpoint_path,
        )

    assert coords == MlflowCoordinates("sqlite:///resolved.db", "mlflow-exp-1", "run-123")
    assert log_context.call_args.kwargs["run_id"] == "run-123"
    assert log_context.call_args.kwargs["tracking_uri"] == "sqlite:///resolved.db"
    assert ensure_checkpoint.call_args.kwargs["checkpoint_path"] == checkpoint_path
    assert log_artifacts.call_args.kwargs["run_id"] == "run-123"


def test_finalize_fallback_mlflow_run_returns_the_freshly_created_run(
    finalization_context: _TrainingFinalizationContext,
) -> None:
    """The fallback branch reports the tracking URI it was configured with, plus the new run."""
    checkpoint_path = Path("ws/checkpoints/model.ckpt")

    with patch(
        "neuralls.composition.assignments.training.create_fallback_training_run",
        return_value=("fallback-exp", "fallback-run"),
    ):
        coords = _finalize_fallback_mlflow_run(
            context=finalization_context,
            checkpoint_path=checkpoint_path,
        )

    assert coords == MlflowCoordinates("sqlite:///fallback.db", "fallback-exp", "fallback-run")


def test_finalize_fallback_mlflow_run_returns_none_when_run_creation_fails(
    finalization_context: _TrainingFinalizationContext,
) -> None:
    """A fallback-run creation failure is swallowed into None, never raised."""
    checkpoint_path = Path("ws/checkpoints/model.ckpt")

    with patch(
        "neuralls.composition.assignments.training.create_fallback_training_run",
        side_effect=RuntimeError("no mlflow server reachable"),
    ):
        coords = _finalize_fallback_mlflow_run(
            context=finalization_context,
            checkpoint_path=checkpoint_path,
        )

    assert coords is None
