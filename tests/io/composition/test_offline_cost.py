"""Offline costs are charged only to validated method data dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.types.split import IndexSplit
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison._offline_cost import (
    resolve_fit_cost,
    resolve_generation_cost,
    resolve_offline_cost,
)
from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    NoFitData,
    PinnedSplit,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
)
from neuralls.domain.solver.models.cost import (
    FitFolds,
    Known,
    NotApplicable,
    OfflineStages,
    TrainShare,
    TrainShareOfGeneration,
    Unavailable,
)
from neuralls.platform.config.models.checkpoint_refs import NeuralCheckpointRef
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetScaling
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.system_scaling import NoScaling
from neuralls.shared.types import CostProvenance
from tests.fakes.mlflow_client import FakeMlflowClient
from tests.fakes.split_store import InMemorySplitStore

GENERATION_SECONDS = 12.5
WHOLE_GENERATION = Known(GENERATION_SECONDS, 1024, CostProvenance.HISTORICAL)
CUBE_SHARE = TrainShare(0.9, FitFolds.TRAIN)
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
            scaling=DatasetScaling(
                scaling=NoScaling(),
                matrix_norm=1.0,
                matrix_norm_type="spectral",
                matrix_scales={0: 1.0},
            ),
            generation_duration_seconds=GENERATION_SECONDS,
            generation_peak_memory_bytes=1024,
        ),
    )
    return directory


def _split(train_rows: int, val_rows: int, test_rows: int) -> IndexSplit:
    rows = iter(range(train_rows + val_rows + test_rows))
    return IndexSplit(
        train=tuple(next(rows) for _ in range(train_rows)),
        validation=tuple(next(rows) for _ in range(val_rows)),
        test=tuple(next(rows) for _ in range(test_rows)),
        predict=None,
    )


def _verified(
    dataset_dir: Path, folds: tuple[int, int, int] = (180, 0, 20), fit: FitFolds = FitFolds.TRAIN
) -> VerifiedTrainingDataset:
    return VerifiedTrainingDataset(
        dataset_dir, "sha256:verified", PinnedSplit(_split(*folds), "sha256:split"), fit
    )


@pytest.fixture
def splits(dataset_dir: Path) -> InMemorySplitStore:
    return InMemorySplitStore({dataset_dir: _split(180, 0, 20)})


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
            scaling=DatasetScaling(
                scaling=NoScaling(),
                matrix_norm=1.0,
                matrix_norm_type="spectral",
                matrix_scales={0: 1.0},
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
    assert resolve_offline_cost(
        NoFitData(), runs=runs, splits=InMemorySplitStore()
    ) == OfflineStages(NotApplicable(), NotApplicable())


def test_inline_fit_charges_its_train_folds_share_of_generation(
    dataset_dir: Path, runs: MlflowClient, splits: InMemorySplitStore
) -> None:
    assert resolve_offline_cost(InlineFit(dataset_dir), runs=runs, splits=splits) == (
        OfflineStages(TrainShareOfGeneration(WHOLE_GENERATION, CUBE_SHARE), NotApplicable())
    )


def test_missing_manifest_duration_is_unavailable(unstamped_dataset: Path) -> None:
    assert resolve_generation_cost(
        InlineFit(unstamped_dataset), InMemorySplitStore()
    ) == Unavailable("no dataset manifest records a generation duration")


def test_missing_manifest_is_unavailable(tmp_path: Path) -> None:
    assert resolve_generation_cost(
        InlineFit(tmp_path / "absent"), InMemorySplitStore()
    ) == Unavailable("no dataset manifest records a generation duration")


def test_verified_checkpoint_charges_attested_dataset(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient, splits: InMemorySplitStore
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased((_ref(run_id),), _verified(dataset_dir))
    assert resolve_offline_cost(origin, runs=runs, splits=splits) == OfflineStages(
        TrainShareOfGeneration(WHOLE_GENERATION, CUBE_SHARE),
        Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )


@pytest.mark.parametrize(
    ("folds", "fit", "share", "charged_seconds"),
    [
        pytest.param((180, 0, 20), FitFolds.TRAIN, 0.9, 11.25, id="pod-train-test"),
        pytest.param((160, 20, 20), FitFolds.TRAIN, 0.8, 10.0, id="pod-train-val-test"),
        pytest.param(
            (160, 20, 20), FitFolds.TRAIN_AND_VALIDATION, 0.9, 11.25, id="neural-train-val-test"
        ),
        pytest.param(
            (180, 0, 20), FitFolds.TRAIN_AND_VALIDATION, 0.9, 11.25, id="neural-train-test"
        ),
        pytest.param((200, 0, 0), FitFolds.TRAIN, 1.0, GENERATION_SECONDS, id="pod-all-train"),
    ],
)
def test_a_checkpoint_is_charged_the_share_of_generation_in_the_folds_it_read(
    dataset_dir: Path,
    folds: tuple[int, int, int],
    fit: FitFolds,
    share: float,
    charged_seconds: float,
) -> None:
    origin = CheckpointBased((_ref("run"),), _verified(dataset_dir, folds, fit))
    expected = TrainShareOfGeneration(WHOLE_GENERATION, TrainShare(share, fit))

    assert resolve_generation_cost(origin, InMemorySplitStore()) == expected
    assert expected.charged().wall_time_seconds == pytest.approx(charged_seconds)


@pytest.mark.parametrize(
    ("pinned", "reason"),
    [
        pytest.param(
            None,
            "the fit dataset has no pinned split to take the train share of",
            id="no-split",
        ),
        pytest.param(
            (0, 0, 200), "the fit dataset's pinned split has no train rows", id="empty-train"
        ),
    ],
)
def test_an_inline_fit_without_a_train_fold_has_no_generation_charge(
    dataset_dir: Path, pinned: tuple[int, int, int] | None, reason: str
) -> None:
    splits = InMemorySplitStore({} if pinned is None else {dataset_dir: _split(*pinned)})

    assert resolve_generation_cost(InlineFit(dataset_dir), splits) == Unavailable(reason)


def test_unverified_checkpoint_never_charges_nearby_dataset(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient, splits: InMemorySplitStore
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased((_ref(run_id),), TrainingDataUnavailable("no dataset identity"))
    assert resolve_offline_cost(origin, runs=runs, splits=splits) == OfflineStages(
        Unavailable("no dataset identity"),
        Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )


def test_multiple_checkpoints_deduplicate_dataset_and_training_run(
    dataset_dir: Path, fake_store: FakeMlflowClient, runs: MlflowClient, splits: InMemorySplitStore
) -> None:
    run_id = _finished_run(fake_store)
    origin = CheckpointBased((_ref(run_id), _ref(run_id)), _verified(dataset_dir))
    assert resolve_offline_cost(origin, runs=runs, splits=splits) == OfflineStages(
        TrainShareOfGeneration(WHOLE_GENERATION, CUBE_SHARE),
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
    assert resolve_fit_cost(origin, runs=runs) == Known(
        TRAINING_SECONDS + 30.0, None, CostProvenance.HISTORICAL
    )


def test_explicit_checkpoint_has_unknown_training_cost(runs: MlflowClient) -> None:
    origin = CheckpointBased((_ref(None),), TrainingDataUnavailable("no training run"))
    assert resolve_fit_cost(origin, runs=runs) == Unavailable(
        "1 of 1 checkpoints have no recorded training run"
    )


def test_missing_training_run_is_unavailable(runs: MlflowClient) -> None:
    origin = CheckpointBased((_ref("gone"),), TrainingDataUnavailable("no dataset identity"))
    assert isinstance(resolve_fit_cost(origin, runs=runs), Unavailable)
