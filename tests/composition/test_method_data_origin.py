"""Method data origin is resolved from the checkpoint's training identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    NoFitData,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
    resolve_method_data_origin,
)
from neuralls.domain.identity import IdentityTag
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


@pytest.fixture
def fit_dir() -> Path:
    return Path("fit-dataset")


@pytest.fixture
def training_tags() -> dict[str, str]:
    return {IdentityTag.COMPONENTS: json.dumps({"dataset": "sha256:abcdef0123456789"})}


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
    )
    assert isinstance(origin, NoFitData)


def test_inline_pod_uses_its_snapshot_dataset(fit_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=fit_dir))
    assert resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: pytest.fail("unexpected run lookup"),
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
    ) == InlineFit(fit_dir)


def test_checkpoint_pod_accepts_matching_recorded_digest(
    fit_dir: Path, training_tags: dict[str, str]
) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=fit_dir,
            checkpoint_path=Path("basis.pt"),
            resolved_run_id="run-1",
            resolved_checkpoint_path=Path("basis.pt"),
        )
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: "sha256:abcdef01234567890123456789abcdef0123456789",
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == VerifiedTrainingDataset(
        fit_dir, "sha256:abcdef01234567890123456789abcdef0123456789"
    )


def test_checkpoint_pod_blocks_mismatched_recorded_digest(
    fit_dir: Path, training_tags: dict[str, str]
) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=fit_dir,
            checkpoint_path=Path("basis.pt"),
            resolved_run_id="run-1",
            resolved_checkpoint_path=Path("basis.pt"),
        )
    )
    with pytest.raises(ValueError, match="training dataset digest mismatch"):
        resolve_method_data_origin(
            cfg,
            run_tags=lambda run_id: training_tags,
            training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
            dataset_digest=lambda path: "sha256:123456789",
        )


def test_checkpoint_pod_without_recorded_digest_is_unavailable(fit_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=fit_dir,
            checkpoint_path=Path("basis.pt"),
            resolved_run_id="run-1",
            resolved_checkpoint_path=Path("basis.pt"),
        )
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: {},
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
    )
    assert isinstance(origin, CheckpointBased)
    assert isinstance(origin.training_data, TrainingDataUnavailable)


@pytest.mark.parametrize("error", [MlflowException("missing run"), OSError("store offline")])
def test_unreadable_training_run_is_unavailable(fit_dir: Path, error: Exception) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=fit_dir, checkpoint_path=Path("basis.pt"), resolved_run_id="run-1"
        )
    )

    def unavailable_run(run_id: str) -> dict[str, str]:
        raise error

    origin = resolve_method_data_origin(
        cfg,
        run_tags=unavailable_run,
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
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
    )
    assert isinstance(origin, CheckpointBased)
    assert len(origin.checkpoint_refs) == 2
    assert origin.training_data == VerifiedTrainingDataset(
        Path("shared-dataset"), "sha256:abcdef0123456789" + "0" * 48
    )


def test_two_checkpoints_on_different_training_digests_have_no_single_dataset(
    two_checkpoint_method: NeuralAMGPreconditionerConfig,
) -> None:
    origin = resolve_method_data_origin(
        two_checkpoint_method,
        run_tags=lambda run_id: {
            IdentityTag.COMPONENTS: json.dumps(
                {"dataset": "sha256:" + ("a" if run_id == "run-p" else "b") * 16}
            )
        },
        training_dataset_dir=lambda path: Path(path.stem),
        dataset_digest=lambda path: "sha256:" + ("a" if path.name == "p" else "b") * 64,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable(
        "checkpoints use different training datasets"
    )


def test_checkpoint_pod_without_local_fit_dataset_is_unavailable(
    fit_dir: Path, training_tags: dict[str, str]
) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=fit_dir,
            checkpoint_path=Path("basis.pt"),
            resolved_run_id="run-1",
        )
    )

    def missing_dataset(path: Path) -> str:
        raise FileNotFoundError(path)

    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=missing_dataset,
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable("training dataset is unavailable")


def test_explicit_pod_checkpoint_does_not_claim_its_configured_dataset(fit_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=fit_dir, checkpoint_path=Path("basis.pt"))
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: pytest.fail("unexpected run lookup"),
        training_dataset_dir=lambda path: pytest.fail("unexpected data config lookup"),
        dataset_digest=lambda path: pytest.fail("unexpected dataset hash"),
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == TrainingDataUnavailable(
        "checkpoint has no recorded training run"
    )


def test_neural_checkpoint_uses_verified_training_dataset(training_tags: dict[str, str]) -> None:
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        architecture="Net",
        data_config_path=Path("training.toml"),
        resolved_run_id="run-1",
        resolved_checkpoint_path=Path("network.pt"),
    )
    origin = resolve_method_data_origin(
        cfg,
        run_tags=lambda run_id: training_tags,
        training_dataset_dir=lambda path: Path("training-dataset"),
        dataset_digest=lambda path: "sha256:abcdef01234567890123456789abcdef0123456789",
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == VerifiedTrainingDataset(
        Path("training-dataset"), "sha256:abcdef01234567890123456789abcdef0123456789"
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
    )
    assert isinstance(origin, CheckpointBased)
    assert origin.training_data == VerifiedTrainingDataset(
        Path("training-dataset"), "sha256:abcdef01234567890123456789abcdef0123456789"
    )
