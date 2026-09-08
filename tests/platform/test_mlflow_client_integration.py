"""Integration coverage for find_successful_run against a real sqlite-backed
MLflow tracking store.

test_mlflow_client.py pins find_successful_run's filter-string construction
against a MagicMock client — it can't catch a break in the real tag
round-trip (MLflow's own search_runs filter evaluation, or dlkit's
has_checkpoint_artifact confirming a real checkpoint artifact). These tests
seed genuine FINISHED runs in a real tracking store and call
find_successful_run for real, exercising the exact mechanism the
training/comparison reuse-check cascade depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.artifact_selection import (
    CHECKPOINT_ARTIFACT_DIR,
    CHECKPOINT_FILE_EXTENSION,
)
from neuralls.platform.tracking.mlflow_client import find_successful_run


@pytest.fixture
def sqlite_tracking_uri(tmp_path: Path) -> str:
    """A real, empty MLflow tracking store backed by a local sqlite file."""
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _seed_finished_run(
    client: MlflowClient,
    *,
    experiment_id: str,
    tmp_path: Path,
    assignment_id: str,
    dataset_hash: str,
    with_checkpoint: bool = True,
) -> str:
    """Create a real FINISHED run tagged and (optionally) checkpointed like a
    completed training run — the exact shape find_successful_run searches for."""
    run = client.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id
    client.set_tag(run_id, "assignment_id", assignment_id)
    client.set_tag(run_id, "dataset_hash", dataset_hash)
    if with_checkpoint:
        checkpoint = tmp_path / f"{run_id}{CHECKPOINT_FILE_EXTENSION}"
        checkpoint.write_bytes(b"fake-checkpoint")
        client.log_artifact(run_id, str(checkpoint), artifact_path=CHECKPOINT_ARTIFACT_DIR)
    client.set_terminated(run_id, "FINISHED")
    return run_id


def test_find_successful_run_returns_seeded_run_with_matching_dataset_hash(
    sqlite_tracking_uri: str, tmp_path: Path
) -> None:
    client = MlflowClient(tracking_uri=sqlite_tracking_uri)
    experiment_id = client.create_experiment("Train")
    run_id = _seed_finished_run(
        client,
        experiment_id=experiment_id,
        tmp_path=tmp_path,
        assignment_id="asn-1",
        dataset_hash="hash-abc",
    )

    result = find_successful_run(
        tracking_uri=sqlite_tracking_uri,
        mlflow_experiment_name="Train",
        assignment_id="asn-1",
        dataset_hash="hash-abc",
    )

    assert result == run_id


def test_find_successful_run_returns_none_when_dataset_hash_changed(
    sqlite_tracking_uri: str, tmp_path: Path
) -> None:
    """Simulates a regenerated dataset: the prior run's dataset_hash tag no
    longer matches, so the reuse check must miss and report None."""
    client = MlflowClient(tracking_uri=sqlite_tracking_uri)
    experiment_id = client.create_experiment("Train")
    _seed_finished_run(
        client,
        experiment_id=experiment_id,
        tmp_path=tmp_path,
        assignment_id="asn-1",
        dataset_hash="hash-old",
    )

    result = find_successful_run(
        tracking_uri=sqlite_tracking_uri,
        mlflow_experiment_name="Train",
        assignment_id="asn-1",
        dataset_hash="hash-new",
    )

    assert result is None


def test_find_successful_run_skips_finished_run_without_checkpoint(
    sqlite_tracking_uri: str, tmp_path: Path
) -> None:
    """A FINISHED run tagged correctly but missing its checkpoint artifact
    (the durability-step race find_successful_run's docstring documents)
    must not be trusted as reusable."""
    client = MlflowClient(tracking_uri=sqlite_tracking_uri)
    experiment_id = client.create_experiment("Train")
    _seed_finished_run(
        client,
        experiment_id=experiment_id,
        tmp_path=tmp_path,
        assignment_id="asn-1",
        dataset_hash="hash-abc",
        with_checkpoint=False,
    )

    result = find_successful_run(
        tracking_uri=sqlite_tracking_uri,
        mlflow_experiment_name="Train",
        assignment_id="asn-1",
        dataset_hash="hash-abc",
    )

    assert result is None
