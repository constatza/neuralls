"""Characterization tests for training MLflow-run/checkpoint resolution helpers.

Pins the observable behavior of the `_training_artifacts` helpers that had no
direct coverage before the training-orchestration refactor:

- `_build_training_run_config` — structured vs. ad-hoc run naming/tags.
- `_resolve_local_training_checkpoint` — the candidate precedence order.
- `_finalize_existing_mlflow_run` / `_finalize_fallback_mlflow_run` — which
  MLflow coordinates come back out of each finalization branch.

These assert *values*, not signatures, so they stay meaningful across the
parameter-object/DTO refactor of the same call chain.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neuralls.composition.assignments._training_artifacts import (
    BEST_CHECKPOINT_ARTIFACT_KEY,
    LAST_CHECKPOINT_ARTIFACT_KEY,
    RETAINED_CHECKPOINTS_DIR_NAME,
    MlflowCoordinates,
    _build_training_run_config,
    _resolve_local_training_checkpoint,
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
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.tracking.mlflow import MlflowPaths, MlflowRunConfig

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
def workspace_root(tmp_path: Path) -> Path:
    """Workspace root directory for run-config construction."""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def registry_identity() -> AssignmentIdentity:
    """Fully registry-backed identity (assignment + dataset + job ids all set)."""
    return AssignmentIdentity(
        assignment_id="exp-1",
        assignment_display_name="Experiment One",
        dataset_registry_id="dataset-1",
        dataset_display_name="Dataset One",
        job_registry_id="job-1",
        job_display_name="Job One",
    )


@pytest.fixture
def adhoc_identity() -> AssignmentIdentity:
    """Ad-hoc identity with no registry ids — the non-structured naming path."""
    return AssignmentIdentity(
        assignment_display_name="Ad Hoc Run",
        dataset_display_name="Dataset One",
    )


@pytest.fixture
def finalization_workspace(tmp_path: Path) -> AssignmentWorkspace:
    """Workspace whose checkpoint/retained directories can be populated per-test."""
    workspace = AssignmentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "ws",
        data_dir=tmp_path / "ws" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    return workspace


@pytest.fixture
def finalization_context(
    finalization_workspace: AssignmentWorkspace,
    tmp_path: Path,
) -> _TrainingFinalizationContext:
    """Minimal finalization context for the two `_finalize_*_mlflow_run` branches."""
    config_path = tmp_path / "job.toml"
    config_path.write_text("")
    return _TrainingFinalizationContext(
        training_result=SimpleNamespace(metrics={}),
        run_config=MlflowRunConfig(
            experiment_name="Training",
            run_name="Experiment One",
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
                job_display_name="Job One",
            )
        ),
        assignment_id="exp-1",
        resolved_assignment_display_name="Experiment One",
        dataset_id="dataset-1",
        resolved_dataset_display_name="Dataset One",
        workflow_settings=SimpleNamespace(data=None),
        contract=default_training_dataset_contract(),
        config_path=config_path,
        resolved_data_config_path=None,
        fallback_tracking_uri="sqlite:///fallback.db",
    )


# ---------------------------------------------------------------------------
# _build_training_run_config
# ---------------------------------------------------------------------------


def test_build_training_run_config_uses_structured_tags_when_all_registry_ids_present(
    registry_identity: AssignmentIdentity,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """A fully registry-backed identity produces the structured training run spec."""
    config = _build_training_run_config(
        identity=registry_identity,
        mlflow_experiment_name="CustomTraining",
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=False,
    )

    assert config.experiment_name == "CustomTraining"
    assert config.run_name == "Experiment One"
    assert config.tags["assignment_id"] == "exp-1"
    assert config.tags["dataset_id"] == "dataset-1"
    assert config.tags["job_id"] == "job-1"
    assert config.paths.tracking_uri == "sqlite:///runtime.db"
    assert config.paths.artifact_uri == "file:///artifacts"
    assert config.workspace_root == workspace_root


def test_build_training_run_config_appends_timestamp_when_requested(
    registry_identity: AssignmentIdentity,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """include_timestamp=True suffixes the display name with ' | <timestamp>'."""
    config = _build_training_run_config(
        identity=registry_identity,
        mlflow_experiment_name=None,
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=True,
    )

    assert config.run_name.startswith("Experiment One | ")
    assert config.run_name != "Experiment One"


def test_build_training_run_config_falls_back_to_untagged_spec_without_registry_ids(
    adhoc_identity: AssignmentIdentity,
    runtime_mlflow_env: dict[str, str],
    workspace_root: Path,
) -> None:
    """Missing registry ids drop to a bare, untagged run config named after the display name."""
    config = _build_training_run_config(
        identity=adhoc_identity,
        mlflow_experiment_name=None,
        runtime_mlflow_env=runtime_mlflow_env,
        workspace_root=workspace_root,
        include_timestamp=False,
    )

    assert config.run_name == "Ad Hoc Run"
    assert dict(config.tags) == {}
    assert config.experiment_name  # config default, not None


# ---------------------------------------------------------------------------
# _resolve_local_training_checkpoint — candidate precedence
# ---------------------------------------------------------------------------


def test_resolve_local_checkpoint_prefers_the_training_results_own_checkpoint_path(
    finalization_workspace: AssignmentWorkspace,
    tmp_path: Path,
) -> None:
    """An existing `checkpoint_path` attribute wins over every other candidate."""
    direct = tmp_path / "direct.ckpt"
    direct.write_text("direct")
    best = tmp_path / "best.ckpt"
    best.write_text("best")
    (finalization_workspace.checkpoint_dir / "workspace.ckpt").write_text("workspace")

    resolved = _resolve_local_training_checkpoint(
        training_result=SimpleNamespace(
            checkpoint_path=direct,
            artifacts={BEST_CHECKPOINT_ARTIFACT_KEY: best},
        ),
        workspace=finalization_workspace,
    )

    assert resolved == direct


def test_resolve_local_checkpoint_skips_nonexistent_candidates_in_order(
    finalization_workspace: AssignmentWorkspace,
    tmp_path: Path,
) -> None:
    """Non-existent higher-priority candidates are skipped; 'best' beats 'last'."""
    last = tmp_path / "last.ckpt"
    last.write_text("last")
    best = tmp_path / "best.ckpt"
    best.write_text("best")

    resolved = _resolve_local_training_checkpoint(
        training_result=SimpleNamespace(
            checkpoint_path=tmp_path / "missing.ckpt",
            artifacts={
                BEST_CHECKPOINT_ARTIFACT_KEY: best,
                LAST_CHECKPOINT_ARTIFACT_KEY: last,
            },
        ),
        workspace=finalization_workspace,
    )

    assert resolved == best


def test_resolve_local_checkpoint_falls_back_to_the_workspace_checkpoint_dir(
    finalization_workspace: AssignmentWorkspace,
) -> None:
    """With no result-supplied candidates, the workspace checkpoint dir is scanned."""
    workspace_checkpoint = finalization_workspace.checkpoint_dir / "epoch.ckpt"
    workspace_checkpoint.write_text("workspace")

    resolved = _resolve_local_training_checkpoint(
        training_result=SimpleNamespace(checkpoint_path=None, artifacts={}),
        workspace=finalization_workspace,
    )

    assert resolved == workspace_checkpoint


def test_resolve_local_checkpoint_falls_back_to_the_retained_checkpoints_dir(
    finalization_workspace: AssignmentWorkspace,
) -> None:
    """An empty workspace checkpoint dir falls through to the retained-checkpoints dir."""
    retained_dir = finalization_workspace.root_dir / RETAINED_CHECKPOINTS_DIR_NAME
    retained_dir.mkdir(parents=True)
    retained = retained_dir / "retained.ckpt"
    retained.write_text("retained")

    resolved = _resolve_local_training_checkpoint(
        training_result=SimpleNamespace(checkpoint_path=None, artifacts={}),
        workspace=finalization_workspace,
    )

    assert resolved == retained


def test_resolve_local_checkpoint_returns_none_when_nothing_exists_locally(
    finalization_workspace: AssignmentWorkspace,
) -> None:
    """No candidate on disk resolves to None rather than raising."""
    resolved = _resolve_local_training_checkpoint(
        training_result=SimpleNamespace(checkpoint_path=None, artifacts=None),
        workspace=finalization_workspace,
    )

    assert resolved is None


# ---------------------------------------------------------------------------
# _finalize_existing_mlflow_run / _finalize_fallback_mlflow_run
# ---------------------------------------------------------------------------


def test_finalize_existing_mlflow_run_returns_the_coordinates_it_was_given(
    finalization_context: _TrainingFinalizationContext,
    tmp_path: Path,
) -> None:
    """The existing-run branch makes the run durable and echoes its own coordinates back."""
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("ckpt")

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
    tmp_path: Path,
) -> None:
    """The fallback branch reports the tracking URI it was configured with, plus the new run."""
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("ckpt")

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
    tmp_path: Path,
) -> None:
    """A fallback-run creation failure is swallowed into None, never raised."""
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("ckpt")

    with patch(
        "neuralls.composition.assignments.training.create_fallback_training_run",
        side_effect=RuntimeError("no mlflow server reachable"),
    ):
        coords = _finalize_fallback_mlflow_run(
            context=finalization_context,
            checkpoint_path=checkpoint_path,
        )

    assert coords is None
