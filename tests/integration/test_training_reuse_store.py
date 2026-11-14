"""Training reuse against a real MLflow store: the identity-tag lookup finds only matching runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from tests.training_reuse_support import (
    EXPERIMENT,
    IDENTITY,
    OTHER_IDENTITY,
    PatchedInputs,
    patched_assignment_inputs,
    run_once,
)

from neuralls.application.models import AssignmentSucceeded
from neuralls.composition.tracking.services import MLFLOW_TRACKING
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.run_names import TrainingPresentation


@dataclass(frozen=True, slots=True)
class TrainingStore:
    """A real store and the id of its empty training experiment."""

    client: MlflowClient
    experiment_id: str


@pytest.fixture
def store(mlflow_tracking_uri: str) -> TrainingStore:
    """A real store holding the empty training experiment."""
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    return TrainingStore(client, client.create_experiment(EXPERIMENT))


@pytest.fixture
def inputs() -> Iterator[PatchedInputs]:
    """Every input but the reuse decision patched; the assignment derives ``identity``."""
    with patched_assignment_inputs(IDENTITY) as patched:
        yield patched


def _seed_finished_run(store: TrainingStore, tmp_path: Path, identity: StageIdentity) -> str:
    client = store.client
    run = client.create_run(store.experiment_id, tags=identity.tags())
    checkpoint = tmp_path / "ckpt.ckpt"
    checkpoint.write_bytes(b"x")
    client.log_artifact(run.info.run_id, str(checkpoint), artifact_path="checkpoints")
    client.set_terminated(run.info.run_id, "FINISHED")
    return run.info.run_id


def test_real_store_reuses_run_with_the_same_identity(
    store: TrainingStore,
    inputs: PatchedInputs,
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """A genuinely FINISHED, tagged, checkpointed run in a real MLflow store is reused."""
    run_id = _seed_finished_run(store, tmp_path, IDENTITY)

    result = run_once(
        training_presentation, tracking=MLFLOW_TRACKING, tracking_uri=store.client.tracking_uri
    )

    assert isinstance(result, AssignmentSucceeded)
    assert result.mlflow_run_id == run_id
    inputs.prepare_training_settings.assert_not_called()


def test_real_store_retrains_when_the_identity_changed(
    store: TrainingStore,
    inputs: PatchedInputs,
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """A run keyed on other inputs (e.g. a since-regenerated dataset) is not reused."""
    _seed_finished_run(store, tmp_path, IDENTITY)
    inputs.derive.return_value = OTHER_IDENTITY

    result = run_once(
        training_presentation, tracking=MLFLOW_TRACKING, tracking_uri=store.client.tracking_uri
    )

    inputs.prepare_training_settings.assert_called_once()
    assert result is inputs.prepare_training_settings.return_value
