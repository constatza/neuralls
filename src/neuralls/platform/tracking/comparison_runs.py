"""MLflow adapter of the comparison-runs port: a session run with one child run per method."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.platform.tracking.method_sink import MLflowBenchmarkMethodSink
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.platform.tracking.run_writer import RunWriter, open_run
from neuralls.platform.tracking.runtime_environment import environment_params
from neuralls.platform.tracking.split_records import SPLITS_ARTIFACT, encode_split_records
from neuralls.shared.runtime_environment import RuntimeEnvironment
from neuralls.shared.types import DatasetId, SystemId


@dataclass(frozen=True, slots=True)
class MethodRecordContext:
    """What every method run persists beside its raw events, the same for the whole session.

    ``datasets`` presents each dataset; ``solver_contexts`` and ``references`` give the
    settings and reference state each planned system was timed under.
    """

    datasets: Mapping[DatasetId, DatasetPresentation]
    solver_contexts: Mapping[SystemId, SolverTimingContext]
    references: Mapping[SystemId, ReferenceSelection]


@dataclass(frozen=True, slots=True)
class MlflowMethodRun:
    """A method's open child run."""

    writer: RunWriter
    run_id: str
    workloads: MLflowBenchmarkMethodSink

    @property
    def failures(self) -> MLflowWorkflowFailureSink:
        """Records a blocking failure of this method into its run."""
        return MLflowWorkflowFailureSink(self.writer, self.run_id)

    def record_configured_facts(self, prepared: PreparedMethodPresentation) -> None:
        """Log every configured fact of the method as stable parameters."""
        self.writer.log_params(self.run_id, encode_method_parameters(prepared))

    def record_method(
        self, workload: MethodWorkload, summary: MethodWorkloadSummary, *, root: Path | None = None
    ) -> None:
        """Persist the reduced workload's event artifacts and searchable scalars."""
        self.workloads.record_method(workload, summary, root=root)


@dataclass(frozen=True, slots=True)
class MlflowSessionRun:
    """A comparison session's open run, under which every method run is created."""

    writer: RunWriter
    experiment_id: str
    run_id: str
    methods: MethodRecordContext

    @property
    def failures(self) -> MLflowWorkflowFailureSink:
        """Records a blocking failure of a batch phase into the session run."""
        return MLflowWorkflowFailureSink(self.writer, self.run_id)

    def record_splits(self, records: Sequence[DatasetSplitRecord]) -> None:
        """Store the split records as the session's ``splits.toml``."""
        self.writer.log_text(
            self.run_id, encode_split_records(records).decode("utf-8"), SPLITS_ARTIFACT
        )

    def record_environment(self, environment: RuntimeEnvironment) -> None:
        """Log the runtime environment as the session's ``env.`` parameters."""
        self.writer.log_params(self.run_id, environment_params(environment))

    @contextmanager
    def method_run(self, run: NamedRun, tags: Mapping[str, str]) -> Generator[MlflowMethodRun]:
        """Open a method's child run of this session."""
        with open_run(
            self.writer, self.experiment_id, run_name_text(run), tags, parent_run_id=self.run_id
        ) as run_id:
            yield MlflowMethodRun(
                self.writer,
                run_id,
                MLflowBenchmarkMethodSink(
                    self.writer,
                    run_id,
                    self.methods.datasets,
                    self.methods.solver_contexts,
                    self.methods.references,
                ),
            )


@dataclass(frozen=True, slots=True)
class MlflowComparisonRuns:
    """Opens comparison session runs in experiment ``experiment_id`` of ``writer``'s store."""

    writer: RunWriter
    experiment_id: str
    methods: MethodRecordContext

    @contextmanager
    def session(self, run: NamedRun, tags: Mapping[str, str]) -> Generator[MlflowSessionRun]:
        """Open a session run."""
        with open_run(self.writer, self.experiment_id, run_name_text(run), tags) as run_id:
            yield MlflowSessionRun(self.writer, self.experiment_id, run_id, self.methods)
