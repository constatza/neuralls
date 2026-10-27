"""MLflow artifact resolution required for eval-only workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices
from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.artifact_access import ArtifactLeaseManager
from neuralls.platform.tracking.artifact_selection import (
    CHECKPOINT_ARTIFACT_DIR,
    select_config_artifact_dir,
    select_split_artifact_path,
)
from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint


@dataclass(frozen=True, slots=True)
class TrainingConfigArtifacts:
    """Resolved config artifacts from a completed training run."""

    config_dir: Path | None

    def find_named(self, fallback: Path) -> Path:
        """Return the resolved config with the fallback basename when present."""
        if self.config_dir is None:
            return fallback
        candidate = self.config_dir / fallback.name
        return candidate if candidate.exists() else fallback


@dataclass(frozen=True, slots=True)
class TrainingEvaluationArtifacts:
    """Resolved artifacts required to evaluate one completed training run."""

    checkpoint_path: Path
    split_file: Path
    split_artifact_path: str
    config_artifacts: TrainingConfigArtifacts


class CheckpointArtifactError(RuntimeError):
    """Base for any checkpoint-artifact problem on a training run.

    Lets callers narrow to "any checkpoint issue" (missing, corrupt, or otherwise
    unresolvable) without enumerating every subclass.
    """


class MissingCheckpointArtifactError(CheckpointArtifactError):
    """Raised when a training run has no checkpoint artifact to evaluate."""


class CorruptCheckpointArtifactError(CheckpointArtifactError):
    """Raised when a run's checkpoint artifact is registered but fails to resolve.

    Distinct from ``MissingCheckpointArtifactError``: the artifact IS listed under
    ``checkpoints/`` (confirmed via ``list_artifacts``), but local resolution or
    remote materialization failed. MLflow's own client can hide the real
    underlying cause at DEBUG log level and raise a generic exception regardless
    of cause; this wraps that exception with a message that states which failure
    mode this is, so it isn't confused with a genuinely absent checkpoint.
    """


def resolve_training_checkpoint(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
    assignment_id: str,
) -> Path:
    """Resolve training checkpoint artifacts and return the selected checkpoint.

    Args:
        client: MLflow client bound to the target tracking URI.
        run_id: FINISHED training run to resolve the checkpoint from.
        artifact_leases: Lease manager that provides local artifact paths.
        assignment_id: Assignment id the run belongs to, for the error message.

    Raises:
        MissingCheckpointArtifactError: If the run has no checkpoint artifact.
        CorruptCheckpointArtifactError: If checkpoint artifacts are listed but
            cannot be resolved to a valid local path.
    """
    if not client.list_artifacts(run_id, CHECKPOINT_ARTIFACT_DIR):
        raise MissingCheckpointArtifactError(
            f"Training run '{run_id}' for assignment '{assignment_id}' has no "
            f"'{CHECKPOINT_ARTIFACT_DIR}/' artifact and cannot be evaluated. This "
            "is typically a stale run trained before dlkit's checkpoint-deletion "
            "fix (dlkit commit 46107a9); the checkpoint is unrecoverable — "
            "retrain this assignment."
        )
    try:
        checkpoint_root = artifact_leases.resolve_dir(run_id, CHECKPOINT_ARTIFACT_DIR).path
    except Exception as exc:
        raise CorruptCheckpointArtifactError(
            f"Training run '{run_id}' for assignment '{assignment_id}' has a "
            f"'{CHECKPOINT_ARTIFACT_DIR}/' artifact listed, but MLflow could not "
            f"resolve it — this is NOT a missing checkpoint. Underlying error: {exc}"
        ) from exc
    return find_single_checkpoint(checkpoint_root)


@dataclass(frozen=True, slots=True)
class TrainingSplitArtifact:
    """Resolved training split file plus its MLflow artifact-relative identity."""

    file: Path
    artifact_path: str


def resolve_training_split_artifact(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
) -> TrainingSplitArtifact:
    """Resolve and validate the single split JSON artifact for a training run."""
    split_artifact_path = select_split_artifact_path(client, run_id=run_id)
    split_file = artifact_leases.resolve_file(run_id, split_artifact_path).path
    load_split_indices(split_file)
    return TrainingSplitArtifact(file=split_file, artifact_path=split_artifact_path)


def resolve_training_config_artifacts(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
) -> TrainingConfigArtifacts:
    """Resolve optional staged config artifacts from a training run."""
    config_artifact_dir = select_config_artifact_dir(client, run_id=run_id)
    if config_artifact_dir is None:
        return TrainingConfigArtifacts(config_dir=None)
    return TrainingConfigArtifacts(
        config_dir=artifact_leases.resolve_dir(run_id, config_artifact_dir).path
    )


def resolve_training_evaluation_artifacts(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
    assignment_id: str,
) -> TrainingEvaluationArtifacts:
    """Resolve all MLflow artifacts required for eval-only execution."""
    split_artifact = resolve_training_split_artifact(
        client=client,
        run_id=run_id,
        artifact_leases=artifact_leases,
    )
    return TrainingEvaluationArtifacts(
        checkpoint_path=resolve_training_checkpoint(
            client=client,
            run_id=run_id,
            artifact_leases=artifact_leases,
            assignment_id=assignment_id,
        ),
        split_file=split_artifact.file,
        split_artifact_path=split_artifact.artifact_path,
        config_artifacts=resolve_training_config_artifacts(
            client=client,
            run_id=run_id,
            artifact_leases=artifact_leases,
        ),
    )
