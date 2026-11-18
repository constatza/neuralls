"""Method data origin is resolved from the checkpoint's training identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dlkit.infrastructure.types.split import IndexSplit
from mlflow.exceptions import MlflowException

from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    NoFitData,
    PinnedSplit,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
    resolve_method_data_origin,
)
from neuralls.domain.identity import IdentityTag
from neuralls.domain.solver.models.cost import FitFolds
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)

SPLIT_DIGEST = "sha256:" + "5" * 64
PINNED = PinnedSplit(
    IndexSplit(train=(0, 1), validation=(2,), test=(3,), predict=None), SPLIT_DIGEST
)


@pytest.fixture
def fit_dir() -> Path:
    return Path("fit-dataset")


@pytest.fixture
def training_tags() -> dict[str, str]:
    return {
        IdentityTag.COMPONENTS: json.dumps(
            {"dataset": "sha256:abcdef0123456789", "split": SPLIT_DIGEST[:23]}
        )
    }


@pytest.fixture
def neural() -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        architecture="Net",
        data_config_path=Path("training.toml"),
        resolved_run_id="run-1",
        resolved_checkpoint_path=Path("network.pt"),
    )


@pytest.fixture
def two_checkpoint_method() -> NeuralAMGPreconditionerConfig:
    return NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            checkpoint_path=Path("p.pt"), resolved_run_id="run-p", data_config_path=Path("p.toml")
        ),
        restriction=NeuralTransferConfig(
            checkpoint_path=Path("r.pt"), resolved_run_id="run-r", data_config_path=Path("r.toml")
        ),
    )


@pytest.mark.parametrize(
    "cfg",
    [
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
        IC0PreconditionerConfig(),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig()),
    ],
)
def test_bare_methods_have_no_fit_data(cfg: object) -> None:
    origin = resolve_method_data_origin(
        cfg,  # type: ignore[arg-type]
        run_tags=lambda run_id: pytest.fail("unexpected run lookup"),
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, NoFitData)


def test_pod_is_fitted_inline_from_its_snapshot_dataset(fit_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=fit_dir))
    assert resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: pytest.fail("unexpected run lookup"),
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
        pinned_split=lambda path: PINNED,
    ) == InlineFit(PODCoarseningConfig(dataset_dir=fit_dir))


def test_checkpoint_blocks_mismatched_recorded_digest(
    neural: NeuralPreconditionerConfig, training_tags: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="training dataset digest mismatch"):
        resolve_method_data_origin(
            neural,
            run_tags=lambda run_id: training_tags,
            training_dataset_dir=lambda path: Path("training-dataset"),
            dataset_digest=lambda path: "sha256:123456789",
            pinned_split=lambda path: PINNED,
        )


@pytest.mark.parametrize("error", [MlflowException("missing run"), OSError("store offline")])
def test_unreadable_training_run_is_unavailable(
    neural: NeuralPreconditionerConfig, error: Exception
) -> None:
    def unavailable_run(run_id: str) -> dict[str, str]:
        raise error

    origin = resolve_method_data_origin(
        neural,
        run_tags=unavailable_run,
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable("training run cannot be read")


def test_two_checkpoints_on_one_training_digest_share_one_dataset(
    two_checkpoint_method: NeuralAMGPreconditionerConfig, training_tags: dict[str, str]
) -> None:
    origin = resolve_method_data_origin(
        two_checkpoint_method,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("shared-dataset"),
        dataset_digest=lambda path: "sha256:abcdef0123456789" + "0" * 48,
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert len(origin.checkpoint_refs) == 2
    assert origin.training_data == VerifiedTrainingDataset(
        Path("shared-dataset"),
        "sha256:abcdef0123456789" + "0" * 48,
        PINNED,
        FitFolds.TRAIN_AND_VALIDATION,
    )


def test_two_checkpoints_on_different_training_digests_have_no_single_dataset(
    two_checkpoint_method: NeuralAMGPreconditionerConfig,
) -> None:
    origin = resolve_method_data_origin(
        two_checkpoint_method,
        run_tags=lambda run_id: {
            IdentityTag.COMPONENTS: json.dumps(
                {
                    "dataset": "sha256:" + ("a" if run_id == "run-p" else "b") * 16,
                    "split": SPLIT_DIGEST[:23],
                }
            )
        },
        training_dataset_dir=lambda path: Path(path.stem),
        dataset_digest=lambda path: "sha256:" + ("a" if path.name == "p" else "b") * 64,
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable(
        "checkpoints use different training datasets"
    )


def test_checkpoint_without_local_training_dataset_is_unavailable(
    neural: NeuralPreconditionerConfig, training_tags: dict[str, str]
) -> None:
    def missing_dataset(path: Path) -> str:
        raise FileNotFoundError(path)

    origin = resolve_method_data_origin(
        neural,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("training-dataset"),
        dataset_digest=missing_dataset,
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable("training dataset is unavailable")


def test_neural_checkpoint_uses_verified_training_dataset(
    neural: NeuralPreconditionerConfig, training_tags: dict[str, str]
) -> None:
    origin = resolve_method_data_origin(
        neural,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("training-dataset"),
        dataset_digest=lambda path: "sha256:abcdef01234567890123456789abcdef0123456789",
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == VerifiedTrainingDataset(
        Path("training-dataset"),
        "sha256:abcdef01234567890123456789abcdef0123456789",
        PINNED,
        FitFolds.TRAIN_AND_VALIDATION,
    )


def test_neural_checkpoint_without_dataset_identity_is_unavailable() -> None:
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        architecture="Net",
        resolved_run_id="run-1",
        resolved_checkpoint_path=Path("network.pt"),
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: {},
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert isinstance(origin.training_data, TrainingDataUnavailable)


def test_neural_pod_parameter_dataset_is_not_training_provenance(
    fit_dir: Path, training_tags: dict[str, str]
) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=fit_dir,
            input_names=("params",),
            data_config_path=Path("training.toml"),
            resolved_run_id="run-1",
            resolved_checkpoint_path=Path("network.pt"),
        )
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("training-dataset"),
        dataset_digest=lambda path: "sha256:abcdef01234567890123456789abcdef0123456789",
        pinned_split=lambda path: PINNED,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == VerifiedTrainingDataset(
        Path("training-dataset"),
        "sha256:abcdef01234567890123456789abcdef0123456789",
        PINNED,
        FitFolds.TRAIN_AND_VALIDATION,
    )


@pytest.mark.parametrize(
    ("pinned", "reason"),
    [
        pytest.param(
            PinnedSplit(PINNED.split, "sha256:" + "6" * 64),
            "training split differs from the pinned split",
            id="re-pinned",
        ),
        pytest.param(None, "training dataset has no pinned split", id="unpinned"),
    ],
)
def test_checkpoint_on_another_split_than_the_pinned_one_is_unavailable(
    neural: NeuralPreconditionerConfig,
    training_tags: dict[str, str],
    pinned: PinnedSplit | None,
    reason: str,
) -> None:
    origin = resolve_method_data_origin(
        neural,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("training-dataset"),
        dataset_digest=lambda path: "sha256:abcdef01234567890123456789abcdef0123456789",
        pinned_split=lambda path: pinned,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable(reason)
