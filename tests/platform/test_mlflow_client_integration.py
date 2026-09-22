"""Integration coverage for MlflowIdentityStore against a real sqlite-backed
MLflow tracking store.

test_mlflow_store.py pins lookup ordering/scoping with a stubbed checkpoint
check. These tests seed genuine FINISHED runs and call the store for real,
including dlkit's has_checkpoint_artifact confirming a real checkpoint
artifact — the exact mechanism the training/comparison reuse cascade depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.domain.identity import Reused, StageIdentity
from neuralls.platform.tracking.artifact_selection import (
    CHECKPOINT_ARTIFACT_DIR,
    CHECKPOINT_FILE_EXTENSION,
)
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore
from neuralls.shared.digest import canonical_digest


@pytest.fixture
def sqlite_tracking_uri(tmp_path: Path) -> str:
    """A real, empty MLflow tracking store backed by a local sqlite file."""
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def identity() -> StageIdentity:
    """Training identity of the seeded run."""
    return StageIdentity.build("training", {"dataset": canonical_digest("old")})


@pytest.fixture
def changed_identity() -> StageIdentity:
    """Identity after the dataset changed."""
    return StageIdentity.build("training", {"dataset": canonical_digest("new")})


def _seed_finished_run(
    client: MlflowClient,
    *,
    experiment_id: str,
    tmp_path: Path,
    identity: StageIdentity,
    with_checkpoint: bool = True,
) -> str:
    """Create a real FINISHED run tagged and (optionally) checkpointed like a
    completed training run — the exact shape the identity store searches for."""
    run = client.create_run(experiment_id=experiment_id, tags=identity.tags())
    run_id = run.info.run_id
    if with_checkpoint:
        checkpoint = tmp_path / f"{run_id}{CHECKPOINT_FILE_EXTENSION}"
        checkpoint.write_bytes(b"fake-checkpoint")
        client.log_artifact(run_id, str(checkpoint), artifact_path=CHECKPOINT_ARTIFACT_DIR)
    client.set_terminated(run_id, "FINISHED")
    return run_id


@pytest.fixture
def store(sqlite_tracking_uri: str) -> MlflowIdentityStore:
    """Store requiring a real checkpoint artifact, as training reuse does."""
    return MlflowIdentityStore(
        tracking_uri=sqlite_tracking_uri, experiment="Train", require_checkpoint=True
    )


@pytest.fixture
def client(sqlite_tracking_uri: str) -> MlflowClient:
    """Client bound to the real store."""
    return MlflowClient(tracking_uri=sqlite_tracking_uri)


def test_finds_seeded_run_with_matching_identity(
    store: MlflowIdentityStore, client: MlflowClient, identity: StageIdentity, tmp_path: Path
) -> None:
    experiment_id = client.create_experiment("Train")
    run_id = _seed_finished_run(
        client, experiment_id=experiment_id, tmp_path=tmp_path, identity=identity
    )

    assert store.find(identity) == Reused(run_id=run_id)


def test_misses_when_the_identity_changed(
    store: MlflowIdentityStore,
    client: MlflowClient,
    identity: StageIdentity,
    changed_identity: StageIdentity,
    tmp_path: Path,
) -> None:
    """Simulates a regenerated dataset: the prior run's key no longer matches."""
    experiment_id = client.create_experiment("Train")
    _seed_finished_run(client, experiment_id=experiment_id, tmp_path=tmp_path, identity=identity)

    assert store.find(changed_identity) is None


def test_skips_finished_run_without_checkpoint(
    store: MlflowIdentityStore, client: MlflowClient, identity: StageIdentity, tmp_path: Path
) -> None:
    """A FINISHED run keyed correctly but missing its checkpoint artifact (the
    durability-step race) must not be trusted as reusable."""
    experiment_id = client.create_experiment("Train")
    _seed_finished_run(
        client,
        experiment_id=experiment_id,
        tmp_path=tmp_path,
        identity=identity,
        with_checkpoint=False,
    )

    assert store.find(identity) is None
