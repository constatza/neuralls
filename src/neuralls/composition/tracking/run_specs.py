"""Pure builders for MLflow run specs and structured tag dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from neuralls.platform.config.models.experiments import AssignmentEntry, ComparisonRegistryEntry
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.mlflow import MlflowRunConfig


def iso_timestamp(dt: datetime | None = None) -> str:
    """Return local ISO 8601 timestamp at second precision (injectable for tests).

    Args:
        dt: Optional datetime to format. Defaults to now().

    Returns:
        ISO 8601 formatted timestamp string at second precision.
    """
    return (dt or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")


def format_run_timestamp(dt: datetime | None = None) -> str:
    """Return a readable local timestamp for MLflow run names."""
    return (dt or datetime.now()).strftime("%a %d %b %Y - %H:%M:%S")


@dataclass(frozen=True)
class TrainingRunTags:
    """Structured tags for a training run.

    Attributes:
        phase: Always "training".
        assignment_id: Stable assignment identifier.
        dataset_id: Dataset registry id.
        job_id: Job registry id.
        assignment_display_name: Human-facing assignment label.
    """

    phase: Literal["training"]
    assignment_id: str
    dataset_id: str
    job_id: str
    assignment_display_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRunTags:
    """Structured tags for a comparison outer run.

    Attributes:
        phase: Always "comparison".
        comparison_id: Stable comparison identifier.
        comparison_display_name: Human-facing comparison label.
        comparison_config: Stem of comparison config file.
        comparison_path: Full path string.
        started_at: ISO timestamp (local).
        run_name: Timestamped run name (for child back-linking).
    """

    phase: Literal["comparison"]
    comparison_id: str
    comparison_display_name: str
    comparison_config: str
    comparison_path: str
    started_at: str
    run_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRunTags:
    """Structured tags for an eval-only run."""

    phase: Literal["evaluation"]
    assignment_id: str
    source_training_run_id: str
    split_artifact: str
    assignment_display_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class ChildComparisonRunTags:
    """Structured tags for a per-preconditioner nested CG run.

    Attributes:
        phase: Always "preconditioner_run".
        preconditioner: Preconditioner name.
        comparison_id: Parent comparison identifier.
        parent_run_name: Timestamped parent run name (for back-linking).
    """

    phase: Literal["preconditioner_run"]
    preconditioner: str
    comparison_id: str
    parent_run_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class SessionRunTags:
    """Structured tags for a batch session parent run (training or comparison).

    Attributes:
        phase: "session_training" or "session_comparison".
        case_config: Stem of the case config file.
        case_config_path: Full path string.
        started_at: ISO timestamp (local).
        experiment_name: MLflow experiment the session's child runs are grouped under.
    """

    phase: Literal["session_training", "session_comparison", "session_evaluation"]
    case_config: str
    case_config_path: str
    started_at: str
    experiment_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class RegistrationTags:
    """Structured tags for a registered model version.

    Note: registered_at UTC timestamp is added as an invariant inside
    register_logged_model() — not here — so callers cannot accidentally omit it.

    Attributes:
        assignment_id: Stable assignment identifier.
        dataset_id: Dataset registry id.
        job_id: Job registry id.
        assignment_display_name: Human-facing assignment label.
        model_class: Optional model class name from [MODEL].name.
    """

    assignment_id: str
    dataset_id: str
    job_id: str
    assignment_display_name: str
    model_class: str | None = None

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict, omitting None values."""
        tags: dict[str, str] = {
            "assignment_id": self.assignment_id,
            "dataset_id": self.dataset_id,
            "job_id": self.job_id,
            "assignment_display_name": self.assignment_display_name,
        }
        if self.model_class is not None:
            tags["model_class"] = self.model_class
        return tags


def build_training_run_spec(
    *,
    entry: AssignmentEntry,
    experiment_name: str,
    paths: MlflowPaths,
    workspace_root: Path,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> MlflowRunConfig:
    """Build complete MlflowRunConfig for a training run.

    Run name format: ``{display_name} | {readable_timestamp}`` when
    ``include_timestamp`` is True (standalone runs), or just ``{display_name}``
    when False (batch runs where the session parent already carries the timestamp).

    Args:
        entry: Assignment registry entry.
        experiment_name: MLflow experiment name to group runs under.
        paths: Resolved MLflow tracking/artifact URIs.
        workspace_root: Workspace root directory.
        timestamp: Optional fixed timestamp (injectable for tests).
        include_timestamp: When False, omit the timestamp from the run name.

    Returns:
        Complete MlflowRunConfig ready for training.
    """
    ts = timestamp or format_run_timestamp()
    tags = TrainingRunTags(
        phase="training",
        assignment_id=entry.id,
        dataset_id=entry.dataset_id,
        job_id=entry.job_id,
        assignment_display_name=entry.effective_display_name,
    )
    display = entry.effective_display_name
    run_name = f"{display} | {ts}" if include_timestamp else display
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags.as_mlflow_tags(),
        paths=paths,
        workspace_root=workspace_root,
    )


def build_comparison_run_spec(
    *,
    entry: ComparisonRegistryEntry,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> tuple[str, ComparisonRunTags]:
    """Build (run_name, ComparisonRunTags) for a comparison outer run.

    Returns structured tags — caller calls ``.as_mlflow_tags()`` at the MLflow boundary.

    Run name format: ``{display_name} | {timestamp}`` when ``include_timestamp``
    is True (standalone runs), or just ``{display_name}`` when False (batch runs
    where the session parent already carries the timestamp).

    Args:
        entry: Comparison registry entry.
        timestamp: Optional fixed timestamp (injectable for tests).
        include_timestamp: When False, omit the timestamp from the run name.

    Returns:
        Tuple of ``(run_name, ComparisonRunTags)``.
    """
    ts = timestamp or iso_timestamp()
    display = entry.effective_display_name
    run_name = f"{display} | {ts}" if include_timestamp else display
    method_label = entry.method.stem if entry.method is not None else entry.id
    tags = ComparisonRunTags(
        phase="comparison",
        comparison_id=entry.id,
        comparison_display_name=entry.effective_display_name,
        comparison_config=method_label,
        comparison_path=entry.method.as_posix() if entry.method is not None else entry.id,
        started_at=ts,
        run_name=run_name,
    )
    return run_name, tags


def build_session_run_spec(
    *,
    case_config_path: Path,
    experiment_name: str,
    phase: Literal["session_training", "session_comparison", "session_evaluation"],
    timestamp: str | None = None,
) -> tuple[str, SessionRunTags]:
    """Build (run_name, SessionRunTags) for one batch session invocation.

    Shared by training and comparison batches: both wrap their per-entry runs
    under one ``"{case_config_path.stem} | {timestamp}"`` parent run.

    Args:
        case_config_path: Path to the case config driving this batch.
        experiment_name: MLflow experiment the session's child runs belong to.
        phase: Which batch kind this session wraps.
        timestamp: Optional fixed timestamp (injectable for tests).

    Returns:
        Tuple of ``(run_name, SessionRunTags)``.
    """
    ts = timestamp or iso_timestamp()
    run_name = f"{case_config_path.stem} | {ts}"
    tags = SessionRunTags(
        phase=phase,
        case_config=case_config_path.stem,
        case_config_path=case_config_path.as_posix(),
        started_at=ts,
        experiment_name=experiment_name,
    )
    return run_name, tags


def build_evaluation_tags(
    *,
    assignment_id: str,
    assignment_display_name: str,
    source_training_run_id: str,
    split_artifact: str,
) -> EvaluationRunTags:
    """Build structured tags for a checkpoint evaluation run."""
    return EvaluationRunTags(
        phase="evaluation",
        assignment_id=assignment_id,
        source_training_run_id=source_training_run_id,
        split_artifact=split_artifact,
        assignment_display_name=assignment_display_name,
    )


def build_child_comparison_tags(
    *,
    preconditioner_name: str,
    comparison_id: str,
    parent_run_name: str,
) -> ChildComparisonRunTags:
    """Build structured tags for a nested per-preconditioner CG run.

    Args:
        preconditioner_name: Name of the preconditioner.
        comparison_id: Parent comparison identifier.
        parent_run_name: Timestamped parent run name for back-linking.

    Returns:
        ChildComparisonRunTags with all fields set.
    """
    return ChildComparisonRunTags(
        phase="preconditioner_run",
        preconditioner=preconditioner_name,
        comparison_id=comparison_id,
        parent_run_name=parent_run_name,
    )


def build_registration_tags(
    *,
    entry: AssignmentEntry,
    model_class: str | None,
) -> RegistrationTags:
    """Build structured registration tags for a model version.

    Args:
        entry: Assignment registry entry.
        model_class: Optional model class name from [MODEL].name.

    Returns:
        RegistrationTags with all available fields populated.
    """
    return RegistrationTags(
        assignment_id=entry.id,
        dataset_id=entry.dataset_id,
        job_id=entry.job_id,
        assignment_display_name=entry.effective_display_name,
        model_class=model_class,
    )
