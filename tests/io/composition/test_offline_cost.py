"""Offline costs are charged only to validated method data dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison._offline_cost import (
    resolve_generation_cost,
    resolve_offline_cost,
    resolve_training_cost,
)
from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    NoFitData,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
)
from neuralls.domain.solver.models.cost import Known, NotApplicable, OfflineStages, Unavailable
from neuralls.platform.config.models.checkpoint_refs import NeuralCheckpointRef
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.types import CostProvenance
from tests.fakes.mlflow_client import FakeMlflowClient

GENERATION_SECONDS = 12.5
TRAINING_SECONDS = 90.5
START_MS = 10_000


@pytest.fixture
def fake_store(tmp_path: Path) -> FakeMlflowClient:
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


@pytest.fixture
def runs(fake_store: FakeMlflowClient) -> MlflowClient:
    return MlflowTrackingServices(client_factory=fake_store).client(fake_store.tracking_uri)


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "fit-data"
    directory.mkdir()
    artifact = DatasetArtifact(path="matrix.npy", format="npy", dtype="float64", shape=(10, 10))
    vector = DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(10,))
    save_dataset_manifest(
        directory,
        make_dataset_manifest(
            matrix=artifact,
            rhs=vector,
            solutions=vector,
            normalization=DatasetNormalization(
                type="none", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
            ),
            generation_duration_seconds=GENERATION_SECONDS,
            generation_peak_memory_bytes=1024,
        ),
    )
    return directory


@pytest.fixture
def unstamped_dataset(tmp_path: Path) -> Path:
    directory = tmp_path / "old-data"
    directory.mkdir()
    artifact = DatasetArtifact(path="matrix.npy", format="npy", dtype="float64", shape=(10, 10))
    vector = DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(10,))
    save_dataset_manifest(
        directory,
        make_dataset_manifest(
            matrix=artifact,
            rhs=vector,
            solutions=vector,
            normalization=DatasetNormalization(
                type="none", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
            ),
            generation_duration_seconds=None,
            generation_peak_memory_bytes=None,
        ),
    )
    return directory


def _ref(run_id: str | None) -> NeuralCheckpointRef:
    return NeuralCheckpointRef(checkpoint_path=Path("model.ckpt"), resolved_run_id=run_id)


def _finished_run(store: FakeMlflowClient, seconds: float = TRAINING_SECONDS) -> str:
    return store.add_run(
        "trained", start_time_ms=START_MS, end_time_ms=START_MS + round(seconds * 1000)
    )


def test_no_fit_data_has_no_offline_cost(runs: MlflowClient) -> None:
    assert resolve_offline_cost(NoFitData(), runs=runs) == OfflineStages(
        NotApplicable(), NotApplicable()
    )


def test_inline_fit_charges_only_generation(dataset_dir: Path, runs: MlflowClient) -> None:
    assert resolve_offline_cost(InlineFit(dataset_dir), runs=runs) == OfflineStages(
        Known(GENERATION_SECONDS, 1024, CostProvenance.HISTORICAL), NotApplicable()
    )


def test_missing_manifest_duration_is_unavailable(unstamped_dataset: Path) -> None:
    assert resolve_generation_cost(InlineFit(unstamped_dataset)) == Unavailable(
        "no dataset manifest records a generation duration"
    )


def test_missing_manifest_is_unavailable(tmp_path: Path) -> None:
    assert resolve_generation_cost(InlineFit(tmp_path / "absent")) == Unavailable(
        "no dataset manifest records a generation duration"
    )


def test_verified_checkpoint_charges_attested_dataset(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased(
        (_ref(run_id),), VerifiedTrainingDataset(dataset_dir, "sha256:verified")
    )
    assert resolve_offline_cost(origin, runs=runs) == OfflineStages(
        Known(GENERATION_SECONDS, 1024, CostProvenance.HISTORICAL),
        Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )


def test_unverified_checkpoint_never_charges_nearby_dataset(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased((_ref(run_id),), TrainingDataUnavailable("no dataset identity"))
    assert resolve_offline_cost(origin, runs=runs) == OfflineStages(
        Unavailable("no dataset identity"),
        Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )


def test_multiple_checkpoints_deduplicate_dataset_and_training_run(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased(
        (_ref(run_id), _ref(run_id)),
        VerifiedTrainingDataset(dataset_dir, "sha256:verified"),
    )
    assert resolve_offline_cost(origin, runs=runs) == OfflineStages(
        Known(GENERATION_SECONDS, 1024, CostProvenance.HISTORICAL),
        Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )


def test_distinct_training_runs_add_their_wall_times(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    first = _finished_run(fake_store)
    second = _finished_run(fake_store, 30.0)
    origin = CheckpointBased(
        (_ref(first), _ref(second)), TrainingDataUnavailable("different datasets")
    )
    assert resolve_training_cost(origin, runs=runs) == Known(
        TRAINING_SECONDS + 30.0, None, CostProvenance.HISTORICAL
    )


def test_explicit_checkpoint_has_unknown_training_cost(runs: MlflowClient) -> None:
    origin = CheckpointBased((_ref(None),), TrainingDataUnavailable("no training run"))
    assert resolve_training_cost(origin, runs=runs) == Unavailable(
        "1 of 1 checkpoints have no recorded training run"
    )


def test_missing_training_run_is_unavailable(runs: MlflowClient) -> None:
    origin = CheckpointBased((_ref("gone"),), TrainingDataUnavailable("no dataset identity"))
    assert isinstance(resolve_training_cost(origin, runs=runs), Unavailable)
