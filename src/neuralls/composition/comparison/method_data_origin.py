"""Validated data origin of a resolved comparison method."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from mlflow.exceptions import MlflowException

from neuralls.domain.identity import IdentityTag
from neuralls.platform.config.models.checkpoint_refs import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.shared.digest import Digest
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
    """A POD basis fitted from snapshots during each matrix setup."""

    fit_dataset: Path


@dataclass(frozen=True, slots=True)
class VerifiedTrainingDataset:
    """A dataset whose current digest matches the training run's identity."""

    dataset: Path
    digest: Digest


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


def _recorded_digest(tags: Mapping[str, str]) -> str | None:
    """Read the training dataset identity; older or external runs may lack it."""
    try:
        components = json.loads(tags[IdentityTag.COMPONENTS])
        digest = components["dataset"]
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
) -> MethodDataOrigin:
    """Resolve method fit data from checkpoint identity and current dataset content.

    A POD checkpoint's configured snapshot directory must match any recorded
    training digest. For other checkpoints, the training data config locates
    the dataset; a nearby runtime parameter directory is never evidence.
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
        return InlineFit(Path(pod.dataset_dir)) if pod is not None else NoFitData()

    pod_dir = Path(pod.dataset_dir) if pod is not None else None
    checks = [
        _verify_checkpoint_data(
            ref,
            pod_dir=pod_dir,
            run_tags=run_tags,
            training_dataset_dir=training_dataset_dir,
            dataset_digest=dataset_digest,
        )
        for ref in refs
    ]
    verified = [check for check in checks if isinstance(check, VerifiedTrainingDataset)]
    unavailable = [check.reason for check in checks if isinstance(check, _Unverified)]
    if unavailable:
        return CheckpointBased(refs, TrainingDataUnavailable("; ".join(dict.fromkeys(unavailable))))
    if len({item.digest for item in verified}) != 1:
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
    run_tags: Callable[[str], Mapping[str, str]],
    training_dataset_dir: Callable[[Path], Path],
    dataset_digest: Callable[[Path], Digest],
) -> VerifiedTrainingDataset | _Unverified:
    """Check one checkpoint's training dataset against the digest its run recorded.

    Raises:
        ValueError: The dataset exists but its digest differs from the recorded one.
    """
    if ref.resolved_run_id is None:
        return _Unverified("checkpoint has no recorded training run")
    try:
        recorded = _recorded_digest(run_tags(ref.resolved_run_id))
    except MlflowException, OSError:
        return _Unverified("training run cannot be read")
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
    return VerifiedTrainingDataset(directory, current)


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
