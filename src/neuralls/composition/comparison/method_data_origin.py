"""Validated data origin of a resolved comparison method."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from dlkit.infrastructure.types.split import IndexSplit
from mlflow.exceptions import MlflowException

from neuralls.composition.generation.ports import SplitStore
from neuralls.domain.identity import IdentityTag
from neuralls.domain.solver.models.cost import FitFolds
from neuralls.platform.config.models.checkpoint_refs import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.shared.digest import Digest, content_digest
from neuralls.shared.types import (
    InlineFitEvidence,
    MethodDataEvidence,
    NoFitDataEvidence,
    UnavailableCheckpointDataEvidence,
    VerifiedCheckpointDataEvidence,
)


@dataclass(frozen=True, slots=True)
class NoFitData:
    """A method built only from the comparison matrix."""


@dataclass(frozen=True, slots=True)
class InlineFit:
    """A POD basis fitted from snapshots during each matrix setup.

    Carries the coarsening it fits, so the dataset and the snapshot cut that decide the
    rows it reads are never re-derived from the method config.
    """

    coarsening: PODCoarseningConfig

    @property
    def fit_dataset(self) -> Path:
        """The dataset whose snapshots the fit reads."""
        return Path(self.coarsening.dataset_dir)


@dataclass(frozen=True, slots=True)
class PinnedSplit:
    """A dataset's pinned split and the digest of the file it is pinned in."""

    split: IndexSplit
    digest: Digest


def read_pinned_split(dataset_dir: Path, store: SplitStore) -> PinnedSplit | None:
    """The split pinned for ``dataset_dir`` with its file digest; ``None`` when none is pinned."""
    split = store.read(dataset_dir)
    if split is None:
        return None
    return PinnedSplit(split, content_digest(store.require(dataset_dir)))


def fold_rows(split: IndexSplit, folds: FitFolds) -> int:
    """The number of dataset rows in ``folds`` of ``split``: the rows a fit of those folds read."""
    match folds:
        case FitFolds.TRAIN:
            return len(split.train)
        case FitFolds.TRAIN_AND_VALIDATION:
            return len(split.train) + len(split.validation)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class VerifiedTrainingDataset:
    """A dataset whose current digest and pinned split match the training run's identity.

    The checkpoint was fit on the ``folds`` of ``split``, the split its run recorded.
    """

    dataset: Path
    digest: Digest
    split: PinnedSplit
    folds: FitFolds


@dataclass(frozen=True, slots=True)
class TrainingDataUnavailable:
    """A checkpoint with no verifiable single training dataset."""

    reason: str


type CheckpointTrainingProvenance = VerifiedTrainingDataset | TrainingDataUnavailable


@dataclass(frozen=True, slots=True)
class CheckpointBased:
    """A method built from resolved checkpoint refs and their training evidence."""

    checkpoint_refs: tuple[NeuralCheckpointRef, ...]
    training_data: CheckpointTrainingProvenance

    def __post_init__(self) -> None:
        """Reject a checkpoint origin without a checkpoint dependency."""
        if not self.checkpoint_refs:
            raise ValueError("checkpoint origin needs at least one resolved reference")


type MethodDataOrigin = NoFitData | InlineFit | CheckpointBased


def origin_evidence(origin: MethodDataOrigin) -> MethodDataEvidence:
    """Project a resolved origin to the path-free evidence stored with a method run."""
    match origin:
        case NoFitData():
            return NoFitDataEvidence()
        case InlineFit():
            return InlineFitEvidence()
        case CheckpointBased(training_data=training_data):
            match training_data:
                case VerifiedTrainingDataset(digest=digest):
                    return VerifiedCheckpointDataEvidence(digest)
                case TrainingDataUnavailable(reason=reason):
                    return UnavailableCheckpointDataEvidence(reason)
                case _ as unreachable:
                    assert_never(unreachable)
        case _ as unreachable:
            assert_never(unreachable)


def _recorded_digest(tags: Mapping[str, str], component: str) -> str | None:
    """Read one training identity component; older or external runs may lack it."""
    try:
        components = json.loads(tags[IdentityTag.COMPONENTS])
        digest = components[component]
    except KeyError, TypeError, ValueError:
        return None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None
    hex_part = digest.removeprefix("sha256:")
    if len(hex_part) < 16 or any(char not in "0123456789abcdef" for char in hex_part):
        return None
    return digest


def resolve_method_data_origin(
    cfg: PreconditionerConfig,
    *,
    run_tags: Callable[[str], Mapping[str, str]],
    training_dataset_dir: Callable[[Path], Path],
    dataset_digest: Callable[[Path], Digest],
    pinned_split: Callable[[Path], PinnedSplit | None],
) -> MethodDataOrigin:
    """Resolve method fit data from checkpoint identity and current dataset content.

    A POD checkpoint's configured snapshot directory must match any recorded
    training digest. For other checkpoints, the training data config locates
    the dataset; a nearby runtime parameter directory is never evidence. A
    checkpoint is verified only while its dataset's pinned split is the one its
    run recorded, since the split decides which rows it was fit on: a POD fit
    job reads the train fold, a neural training job the train and validation folds.
    """
    pod = (
        cfg.coarsening
        if isinstance(cfg, AMGPreconditionerConfig)
        and isinstance(cfg.coarsening, PODCoarseningConfig)
        else None
    )
    refs = (
        tuple(ref for _, ref in cfg.checkpoint_refs())
        if isinstance(cfg, CheckpointRefBearing)
        else ()
    )
    if not refs:
        return InlineFit(pod) if pod is not None else NoFitData()

    pod_dir = Path(pod.dataset_dir) if pod is not None else None
    folds = FitFolds.TRAIN if pod is not None else FitFolds.TRAIN_AND_VALIDATION
    checks = [
        _verify_checkpoint_data(
            ref,
            pod_dir=pod_dir,
            folds=folds,
            run_tags=run_tags,
            training_dataset_dir=training_dataset_dir,
            dataset_digest=dataset_digest,
            pinned_split=pinned_split,
        )
        for ref in refs
    ]
    verified = [check for check in checks if isinstance(check, VerifiedTrainingDataset)]
    unavailable = [check.reason for check in checks if isinstance(check, _Unverified)]
    if unavailable:
        return CheckpointBased(refs, TrainingDataUnavailable("; ".join(dict.fromkeys(unavailable))))
    if len({(item.digest, item.split.digest) for item in verified}) != 1:
        return CheckpointBased(
            refs, TrainingDataUnavailable("checkpoints use different training datasets")
        )
    return CheckpointBased(refs, verified[0])


@dataclass(frozen=True, slots=True)
class _Unverified:
    """Why one checkpoint's training data could not be checked."""

    reason: str


def _verify_checkpoint_data(
    ref: NeuralCheckpointRef,
    *,
    pod_dir: Path | None,
    folds: FitFolds,
    run_tags: Callable[[str], Mapping[str, str]],
    training_dataset_dir: Callable[[Path], Path],
    dataset_digest: Callable[[Path], Digest],
    pinned_split: Callable[[Path], PinnedSplit | None],
) -> VerifiedTrainingDataset | _Unverified:
    """Check one checkpoint's training dataset and split against the digests its run recorded.

    Re-pinning a split leaves the dataset digest unchanged, so a split that differs from the
    recorded one is unverified rather than charged a share of rows the fit never read.

    Raises:
        ValueError: The dataset exists but its digest differs from the recorded one.
    """
    if ref.resolved_run_id is None:
        return _Unverified("checkpoint has no recorded training run")
    try:
        tags = run_tags(ref.resolved_run_id)
    except MlflowException, OSError:
        return _Unverified("training run cannot be read")
    recorded = _recorded_digest(tags, "dataset")
    if recorded is None:
        return _Unverified("training run has no dataset identity")
    directory = _training_directory(ref, pod_dir, training_dataset_dir)
    if isinstance(directory, _Unverified):
        return directory
    try:
        current = dataset_digest(directory)
    except FileNotFoundError:
        return _Unverified("training dataset is unavailable")
    if not current.startswith(recorded):
        raise ValueError(
            f"training dataset digest mismatch for checkpoint run {ref.resolved_run_id!r}"
        )
    recorded_split = _recorded_digest(tags, "split")
    if recorded_split is None:
        return _Unverified("training run has no split identity")
    split = pinned_split(directory)
    if split is None:
        return _Unverified("training dataset has no pinned split")
    if not split.digest.startswith(recorded_split):
        return _Unverified("training split differs from the pinned split")
    return VerifiedTrainingDataset(directory, current, split, folds)


def _training_directory(
    ref: NeuralCheckpointRef,
    pod_dir: Path | None,
    training_dataset_dir: Callable[[Path], Path],
) -> Path | _Unverified:
    """The POD snapshot directory, else the dataset the training data config names."""
    if pod_dir is not None:
        return pod_dir
    if ref.data_config_path is None:
        return _Unverified("training dataset path is unavailable")
    try:
        return training_dataset_dir(ref.data_config_path)
    except OSError, ValueError:
        return _Unverified("training dataset config is unavailable or invalid")
