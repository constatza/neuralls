"""Training checkpoint resolution picks the right checkpoint file on disk."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neuralls.composition.assignments._training_artifacts import (
    BEST_CHECKPOINT_ARTIFACT_KEY,
    LAST_CHECKPOINT_ARTIFACT_KEY,
    RETAINED_CHECKPOINTS_DIR_NAME,
    _resolve_local_training_checkpoint,
)
from neuralls.composition.assignments.training import _resolve_training_checkpoint
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from tests.fakes.checkpoint_lease_manager import CheckpointLeaseManager


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


def test_resolve_training_checkpoint_leases_mlflow_artifacts(
    tmp_path: Path, finalization_workspace: AssignmentWorkspace
) -> None:
    """MLflow fallback checkpoint resolution must not create workspace downloads."""
    workspace = finalization_workspace
    training_result = SimpleNamespace(checkpoint_path=None, artifacts={})
    checkpoint_root = tmp_path / "mlflow-artifacts" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    resolved_checkpoint = checkpoint_root / "model.ckpt"
    resolved_checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_leases = CheckpointLeaseManager(checkpoint_root)

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


def test_resolve_local_checkpoint_prefers_the_training_results_own_checkpoint_path(
    finalization_workspace: AssignmentWorkspace,
    tmp_path: Path,
) -> None:
    """An existing `checkpoint_path` attribute wins over every other candidate."""
    direct = tmp_path / "direct.ckpt"
    direct.write_text("direct", encoding="utf-8")
    best = tmp_path / "best.ckpt"
    best.write_text("best", encoding="utf-8")
    (finalization_workspace.checkpoint_dir / "workspace.ckpt").write_text(
        "workspace", encoding="utf-8"
    )

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
    last.write_text("last", encoding="utf-8")
    best = tmp_path / "best.ckpt"
    best.write_text("best", encoding="utf-8")

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
    workspace_checkpoint.write_text("workspace", encoding="utf-8")

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
    retained.write_text("retained", encoding="utf-8")

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
