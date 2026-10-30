"""Pure builders for MLflow run specs and structured tag dataclasses.

Every run name is a typed ``RunName`` paired with the case's notation (a ``NamedRun``);
the platform tracking adapter renders it when the run opens. Tags keep machine ids in
``*_id`` fields and rendered text in ``*_label`` fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, assert_never

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    ComparisonRunName,
    ComparisonSubject,
    EditorialComparison,
    EvaluationRunName,
    HeldOutMatrixPart,
    MethodRunName,
    NamedRun,
    RhsDrawPart,
    RunStamp,
    RunTimestamp,
    SessionRunName,
    SystemQualifier,
    TrainingPresentation,
    TrainingRunName,
    TrainingSubject,
    Untimed,
)
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
    SessionTitle,
)
from neuralls.domain.solver.models.provenance import HeldOutMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.mlflow import MlflowRunConfig
from neuralls.platform.tracking.run_names import (
    render_session_title,
    render_training_dataset,
    render_training_subject,
    run_name_text,
)
from neuralls.shared.digest import Digest, ordered_workload_digest
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName


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


def _stamp(timestamp: str | None, *, include_timestamp: bool) -> RunStamp:
    """The run's stamp: its timestamp, or none when a session parent already carries one."""
    if not include_timestamp:
        return Untimed()
    return RunTimestamp(timestamp or format_run_timestamp())


def system_qualifier(planned: PlannedSystem) -> SystemQualifier:
    """What tells ``planned`` apart from its sibling systems in a run name.

    A title is shared by every sibling system, so the held-out matrix and the
    right-hand-side draw carry the distinction.
    """
    matrix = (
        (HeldOutMatrixPart(planned.matrix.matrix_id),)
        if isinstance(planned.provenance, HeldOutMatrix)
        else ()
    )
    draw = (RhsDrawPart(planned.rhs_ordinal),) if planned.entry.rhs_count > 1 else ()
    return SystemQualifier((*matrix, *draw))


def training_run_name(
    presentation: TrainingPresentation,
    *,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> NamedRun:
    """The typed name of a training run: what it trains, plus a stamp when it runs standalone."""
    stamp = _stamp(timestamp, include_timestamp=include_timestamp)
    return NamedRun(TrainingRunName(presentation.subject, stamp), presentation.notation)


@dataclass(frozen=True, slots=True)
class TrainingRunTags:
    """Structured tags for a training run.

    ``subject`` is what was trained; its rendered text becomes the ``assignment_label`` tag
    and the dataset it trained on becomes the ``dataset`` tag. ``dataset_id`` stays the
    machine id so runs remain filterable by it.
    """

    phase: Literal["training"]
    assignment_id: str
    dataset_id: str
    job_id: str
    subject: TrainingSubject
    notation: NotationPolicy

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return {
            "phase": self.phase,
            "assignment_id": self.assignment_id,
            "dataset_id": self.dataset_id,
            "dataset": render_training_dataset(self.subject, self.notation),
            "job_id": self.job_id,
            "assignment_label": render_training_subject(self.subject, self.notation),
        }


@dataclass(frozen=True, slots=True)
class ComparisonRunTags:
    """Structured tags for a comparison outer run.

    Attributes:
        phase: Always "comparison".
        comparison_id: Stable comparison identifier.
        title: What the comparison ran on; rendered as the ``comparison_label`` tag.
        comparison_config: Stem of comparison config file.
        comparison_path: Full path string.
        started_at: ISO timestamp (local).
        run: The run's own typed name; rendered as ``run_name`` for child back-linking.
    """

    phase: Literal["comparison"]
    comparison_id: str
    title: ComparisonTitle
    comparison_config: str
    comparison_path: str
    started_at: str
    run: NamedRun

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return {
            "phase": self.phase,
            "comparison_id": self.comparison_id,
            "comparison_label": render_session_title(self.title, self.run.notation),
            "comparison_config": self.comparison_config,
            "comparison_path": self.comparison_path,
            "started_at": self.started_at,
            "run_name": run_name_text(self.run),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRunTags:
    """Structured tags for an eval-only run."""

    phase: Literal["evaluation"]
    assignment_id: str
    source_training_run_id: str
    split_artifact: str
    subject: TrainingSubject
    notation: NotationPolicy

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return {
            "phase": self.phase,
            "assignment_id": self.assignment_id,
            "source_training_run_id": self.source_training_run_id,
            "split_artifact": self.split_artifact,
            "dataset": render_training_dataset(self.subject, self.notation),
            "assignment_label": render_training_subject(self.subject, self.notation),
        }


@dataclass(frozen=True, slots=True)
class ChildComparisonRunTags:
    """Structured tags for a per-preconditioner nested CG run.

    Attributes:
        phase: Always "preconditioner_run".
        method_id: Machine result key of the preconditioner.
        comparison_id: Parent comparison identifier.
        parent: The parent run's typed name; rendered as ``parent_run_name`` for back-linking.
        method: The method's compact presentation; rendered as the ``method_label`` tag.
    """

    phase: Literal["preconditioner_run"]
    method_id: PreconditionerName
    comparison_id: str
    parent: NamedRun
    method: NamedRun

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return {
            "phase": self.phase,
            "method_id": self.method_id,
            "method_label": run_name_text(self.method),
            "comparison_id": self.comparison_id,
            "parent_run_name": run_name_text(self.parent),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMethodRunTags:
    """Identity tags for one direct method child of a session."""

    phase: Literal["comparison_method"]
    method_id: PreconditionerName
    method_label: str
    method_config_digest: Digest
    workload_membership_digest: Digest
    required_artifacts: str
    identity_key: Digest
    identity_components: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize method-child tags for MLflow."""
        tags = asdict(self)
        tags["neuralls.identity.stage"] = self.phase
        tags["neuralls.identity.key"] = self.identity_key
        tags["neuralls.identity.components"] = self.identity_components
        return tags


def build_benchmark_method_run_spec(
    *,
    prepared: PreparedMethodPresentation,
    notation: NotationPolicy,
    config_digest: Digest,
    system_ids: tuple[str, ...],
    system_identities: tuple[Digest, ...] | None = None,
) -> tuple[NamedRun, BenchmarkMethodRunTags]:
    """Build a deterministic method-child name and reusable-workload identity.

    The name is the method's own compact presentation, so the plot legend and the run
    list show the same thing. ``system_ids`` are machine ids, not enough to identify the
    work: a prepared system identity is paired with each when available.
    """
    membership = ordered_workload_digest(system_ids, system_identities)
    identity = StageIdentity.build(
        "comparison_method",
        {
            "method_config": config_digest,
            "workload_membership": membership,
        },
    )
    named = NamedRun(MethodRunName(prepared.compact), notation)
    tags = BenchmarkMethodRunTags(
        phase="comparison_method",
        method_id=prepared.method_id,
        method_label=run_name_text(named),
        method_config_digest=config_digest,
        workload_membership_digest=membership,
        required_artifacts=(
            "setup_events.parquet,solve_attempts.parquet,summary.toml,"
            "method_config.toml,method_diagnostics.json"
        ),
        identity_key=identity.key,
        identity_components=json.dumps(dict(identity.components), sort_keys=True),
    )
    return named, tags


@dataclass(frozen=True, slots=True)
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


def build_training_run_spec(
    *,
    assignment_id: str,
    dataset_id: str,
    job_id: str,
    presentation: TrainingPresentation,
    experiment_name: str,
    paths: MlflowPaths,
    workspace_root: Path,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> MlflowRunConfig:
    """Build complete MlflowRunConfig for a training run.

    The run is named by what it trains (``subject``) plus a timestamp when it runs
    standalone; batch runs omit the timestamp because the session parent carries it.
    """
    tags = TrainingRunTags(
        phase="training",
        assignment_id=assignment_id,
        dataset_id=dataset_id,
        job_id=job_id,
        subject=presentation.subject,
        notation=presentation.notation,
    )
    return MlflowRunConfig(
        experiment_name=experiment_name,
        name=training_run_name(
            presentation, timestamp=timestamp, include_timestamp=include_timestamp
        ),
        tags=tags.as_mlflow_tags(),
        paths=paths,
        workspace_root=workspace_root,
    )


def comparison_subject(title: ComparisonTitle, dataset: DatasetPresentation) -> ComparisonSubject:
    """What a comparison run is about: an editorial title is followed by its dataset."""
    match title:
        case EditorialPresentation():
            return EditorialComparison(title, dataset)
        case ComparisonScopePresentation():
            return title
        case _:
            assert_never(title)


def comparison_run(
    planned: PlannedSystem,
    title: ComparisonTitle,
    dataset: DatasetPresentation,
    notation: NotationPolicy,
    stamp: RunStamp,
) -> NamedRun:
    """The typed name of one planned system's comparison run."""
    return NamedRun(
        ComparisonRunName(comparison_subject(title, dataset), system_qualifier(planned), stamp),
        notation,
    )


def build_comparison_run_spec(
    *,
    entry: ComparisonRegistryEntry,
    planned: PlannedSystem,
    title: ComparisonTitle,
    dataset: DatasetPresentation,
    notation: NotationPolicy,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> tuple[NamedRun, ComparisonRunTags]:
    """Build (run, ComparisonRunTags) for a comparison outer run.

    Siblings planned from one sweep or one multi-draw entry stay distinguishable through
    ``system_qualifier``; the tags keep the machine entry id apart from the rendered label.
    """
    ts = timestamp or iso_timestamp()
    stamp = RunTimestamp(ts) if include_timestamp else Untimed()
    run = comparison_run(planned, title, dataset, notation, stamp)
    method_label = entry.method.stem if entry.method is not None else entry.id
    tags = ComparisonRunTags(
        phase="comparison",
        comparison_id=entry.id,
        title=title,
        comparison_config=method_label,
        comparison_path=entry.method.as_posix() if entry.method is not None else entry.id,
        started_at=ts,
        run=run,
    )
    return run, tags


def build_session_run_spec(
    *,
    title: SessionTitle,
    notation: NotationPolicy,
    case_config_path: Path,
    experiment_name: str,
    phase: Literal["session_training", "session_comparison", "session_evaluation"],
    timestamp: str | None = None,
) -> tuple[NamedRun, SessionRunTags]:
    """Build (run, SessionRunTags) for one batch session invocation.

    Shared by training, evaluation and comparison batches: all wrap their per-entry runs
    under one parent run named by what the session ran on, plus the launch time.
    """
    ts = timestamp or iso_timestamp()
    run = NamedRun(SessionRunName(title, RunTimestamp(ts)), notation)
    tags = SessionRunTags(
        phase=phase,
        case_config=case_config_path.stem,
        case_config_path=case_config_path.as_posix(),
        started_at=ts,
        experiment_name=experiment_name,
    )
    return run, tags


def build_evaluation_run(presentation: TrainingPresentation) -> NamedRun:
    """The typed name of a checkpoint evaluation run."""
    return NamedRun(EvaluationRunName(presentation.subject), presentation.notation)


def build_evaluation_tags(
    *,
    assignment_id: str,
    presentation: TrainingPresentation,
    source_training_run_id: str,
    split_artifact: str,
) -> EvaluationRunTags:
    """Build structured tags for a checkpoint evaluation run."""
    return EvaluationRunTags(
        phase="evaluation",
        assignment_id=assignment_id,
        source_training_run_id=source_training_run_id,
        split_artifact=split_artifact,
        subject=presentation.subject,
        notation=presentation.notation,
    )


def build_child_comparison_tags(
    *,
    prepared: PreparedMethodPresentation,
    notation: NotationPolicy,
    comparison_id: str,
    parent: NamedRun,
) -> ChildComparisonRunTags:
    """Build structured tags for a nested per-preconditioner CG run."""
    return ChildComparisonRunTags(
        phase="preconditioner_run",
        method_id=prepared.method_id,
        comparison_id=comparison_id,
        parent=parent,
        method=NamedRun(MethodRunName(prepared.compact), notation),
    )
