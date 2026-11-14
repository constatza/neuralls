"""In-memory comparison-runs port: records what a comparison session tracks, writing no file."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mlflow.tracking import MlflowClient

from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.platform.tracking.artifact_access import MlflowArtifactLeaseManager
from neuralls.platform.tracking.comparison_runs import MethodRecordContext
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.runtime_environment import RuntimeEnvironment
from neuralls.shared.workflow_failures import WorkflowFailureContext


@dataclass(slots=True)
class RecordedFailures:
    """The failures recorded into one run."""

    recorded: list[tuple[WorkflowFailureContext, Exception]] = field(default_factory=list)

    def record(self, context: WorkflowFailureContext, exc: Exception) -> None:
        self.recorded.append((context, exc))


@dataclass(slots=True)
class RecordedMethodRun:
    """One method run: its name, tags, configured facts, workloads and failures."""

    run: NamedRun
    tags: dict[str, str]
    failures: RecordedFailures = field(default_factory=RecordedFailures)
    configured: list[PreparedMethodPresentation] = field(default_factory=list)
    workloads: list[tuple[MethodWorkload, MethodWorkloadSummary]] = field(default_factory=list)

    def record_configured_facts(self, prepared: PreparedMethodPresentation) -> None:
        self.configured.append(prepared)

    def record_method(
        self, workload: MethodWorkload, summary: MethodWorkloadSummary, *, root: Path | None = None
    ) -> None:
        self.workloads.append((workload, summary))


@dataclass(slots=True)
class RecordedSession:
    """One session run and the method runs opened under it."""

    run: NamedRun
    tags: dict[str, str]
    failures: RecordedFailures = field(default_factory=RecordedFailures)
    splits: list[Sequence[DatasetSplitRecord]] = field(default_factory=list)
    environments: list[RuntimeEnvironment] = field(default_factory=list)
    methods: list[RecordedMethodRun] = field(default_factory=list)

    def record_splits(self, records: Sequence[DatasetSplitRecord]) -> None:
        self.splits.append(records)

    def record_environment(self, environment: RuntimeEnvironment) -> None:
        self.environments.append(environment)

    @contextmanager
    def method_run(self, run: NamedRun, tags: Mapping[str, str]) -> Generator[RecordedMethodRun]:
        method = RecordedMethodRun(run, dict(tags))
        self.methods.append(method)
        yield method


@dataclass(slots=True)
class RecordingComparisonRuns:
    """Every session opened, in order."""

    sessions: list[RecordedSession] = field(default_factory=list)

    @contextmanager
    def session(self, run: NamedRun, tags: Mapping[str, str]) -> Generator[RecordedSession]:
        session = RecordedSession(run, dict(tags))
        self.sessions.append(session)
        yield session


@dataclass(frozen=True, slots=True)
class RecordingTrackingServices:
    """Tracking services over an in-memory store whose comparison runs are recorded."""

    services: MlflowTrackingServices
    runs: RecordingComparisonRuns = field(default_factory=RecordingComparisonRuns)

    def client(self, tracking_uri: str) -> MlflowClient:
        return self.services.client(tracking_uri)

    def identity_store(
        self, *, tracking_uri: str, experiment: str, require_checkpoint: bool = False
    ) -> MlflowIdentityStore:
        return self.services.identity_store(
            tracking_uri=tracking_uri,
            experiment=experiment,
            require_checkpoint=require_checkpoint,
        )

    def lease_manager(self, tracking_uri: str) -> MlflowArtifactLeaseManager:
        return self.services.lease_manager(tracking_uri)

    def comparison_runs(
        self, tracking_uri: str, experiment_id: str, methods: MethodRecordContext
    ) -> RecordingComparisonRuns:
        return self.runs
