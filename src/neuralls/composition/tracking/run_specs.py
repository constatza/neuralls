"""Pure builders for MLflow run specs and structured tag dataclasses.

Every run name is a typed ``RunName`` paired with the case's notation (a ``NamedRun``);
the platform tracking adapter renders it when the run opens. Tags keep machine ids in
``*_id`` fields and rendered text in ``*_label`` fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, assert_never

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    REPORT_PHASE,
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
    SessionPhase,
    SessionRunName,
    SystemQualifier,
    TrainingPresentation,
    TrainingRunName,
    TrainingSubject,
    Untimed,
    session_stage,
)
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
    SessionTitle,
)
from neuralls.domain.solver.models.provenance import HeldOutMatrix
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.benchmark_events import MethodBenchmarkArtifacts
from neuralls.platform.tracking.mlflow import MlflowRunConfig
from neuralls.platform.tracking.run_names import (
    render_session_title,
    render_training_dataset,
    render_training_subject,
    run_name_text,
)
from neuralls.platform.tracking.run_writer import RUN_DESCRIPTION_TAG
from neuralls.platform.tracking.session_artifacts import METHOD_PHASE
from neuralls.shared.digest import Digest, ordered_workload_digest
from neuralls.shared.runtime_environment import Unreported
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
        METHOD_PHASE,
        {
            "method_config": config_digest,
            "workload_membership": membership,
        },
    )
    named = NamedRun(MethodRunName(prepared.compact), notation)
    tags = BenchmarkMethodRunTags(
        phase=METHOD_PHASE,
        method_id=prepared.method_id,
        method_label=run_name_text(named),
        method_config_digest=config_digest,
        workload_membership_digest=membership,
        required_artifacts=",".join(MethodBenchmarkArtifacts.file_names()),
        identity_key=identity.key,
        identity_components=json.dumps(dict(identity.components), sort_keys=True),
    )
    return named, tags


@dataclass(frozen=True, slots=True)
class SessionRunTags:
    """Structured tags for a session root run (training, evaluation or comparison).

    Attributes:
        phase: The session phase lookups filter on; its stage word is the ``neuralls.stage`` tag.
        case_config_path: Full path string.
        started_at: ISO timestamp (local).
        experiment_name: The config key's experiment name, rendered as the ``neuralls.config`` tag.
        title: What the session ran on; rendered as the ``title`` tag and the run description.
        notation: The case notation the title is rendered with.
    """

    phase: SessionPhase
    case_config_path: str
    started_at: str
    experiment_name: str
    title: SessionTitle
    notation: NotationPolicy

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return {
            **root_run_tags(self.phase, self.experiment_name),
            **title_tags(render_session_title(self.title, self.phase, self.notation)),
            "case_config_path": self.case_config_path,
            "started_at": self.started_at,
        }


CONFIG_TAG = "neuralls.config"
"""Root-run tag holding the case config key, ``<case>/<config>``."""

TITLE_TAG = "title"
"""Root-run tag holding the rendered scientific title."""


def root_run_tags(phase: SessionPhase, config: str) -> dict[str, str]:
    """The tags every root run carries: its lookup phase, config key and stage word."""
    return {"phase": phase, CONFIG_TAG: config, "neuralls.stage": session_stage(phase)}


def title_tags(title: str) -> dict[str, str]:
    """A root run's rendered title, as its ``title`` tag and its MLflow description."""
    return {TITLE_TAG: title, RUN_DESCRIPTION_TAG: title}


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


def build_session_run_spec(
    *,
    title: SessionTitle,
    notation: NotationPolicy,
    case_config_path: Path,
    experiment_name: str,
    phase: SessionPhase,
    revision: str | Unreported,
    started: datetime | None = None,
) -> tuple[NamedRun, SessionRunTags]:
    """Build (run, SessionRunTags) for one session root run.

    Shared by training, evaluation and comparison batches: the root is named by its stage,
    start time and code revision; what it ran on is its ``title`` tag and description.
    """
    started = started or datetime.now()
    run = NamedRun(SessionRunName(session_stage(phase), started, revision), notation)
    tags = SessionRunTags(
        phase=phase,
        case_config_path=case_config_path.as_posix(),
        started_at=iso_timestamp(started),
        experiment_name=experiment_name,
        title=title,
        notation=notation,
    )
    return run, tags


def build_report_run_spec(
    *,
    session_tags: Mapping[str, str],
    notation: NotationPolicy,
    config: str,
    revision: str | Unreported,
    started: datetime | None = None,
) -> tuple[NamedRun, dict[str, str]]:
    """Build (run, root tags) for a report root, the child of the session it renders.

    Named like every session root, with the report's own stage, start and revision; it
    carries the session's rendered title, when the session has one, as its own. Its config
    key is the session's, since it sits in the session's experiment; ``config`` stands in
    only for a session recorded before sessions carried one.
    """
    run = NamedRun(
        SessionRunName(session_stage(REPORT_PHASE), started or datetime.now(), revision), notation
    )
    tags = root_run_tags(REPORT_PHASE, session_tags.get(CONFIG_TAG, config))
    title = session_tags.get(TITLE_TAG)
    return run, tags if title is None else {**tags, **title_tags(title)}


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
