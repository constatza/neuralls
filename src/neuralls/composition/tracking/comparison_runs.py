"""Port for the tracking runs one comparison session writes: the session run and its method runs.

Each run is a value handed down from where it was opened. Whatever records into a run (the
failure sink of a scope, a method's workload) is bound to that run, so nothing has to find
"the current run" in process-wide state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol

from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.domain.solver.ports import BenchmarkMethodSink
from neuralls.shared.runtime_environment import RuntimeEnvironment
from neuralls.shared.workflow_failures import WorkflowFailureSink


class MethodRun(BenchmarkMethodSink, Protocol):
    """One method's open run: its configured facts, its reduced workload and its failures."""

    @property
    def failures(self) -> WorkflowFailureSink:
        """Records a blocking failure of this method into its run."""
        ...

    def record_configured_facts(self, prepared: PreparedMethodPresentation) -> None:
        """Record every configured fact of the method, before it executes."""
        ...


class SessionRun(Protocol):
    """The open run of one comparison session."""

    @property
    def failures(self) -> WorkflowFailureSink:
        """Records a blocking failure of a batch phase into the session run."""
        ...

    def record_splits(self, records: Sequence[DatasetSplitRecord]) -> None:
        """Record the held-out systems of every dataset the session ran on."""
        ...

    def record_environment(self, environment: RuntimeEnvironment) -> None:
        """Record the device, CPU, versions and revisions the session was measured on."""
        ...

    def method_run(
        self, run: NamedRun, tags: Mapping[str, str]
    ) -> AbstractContextManager[MethodRun]:
        """Open a method's child run; it finishes, or fails, with the block."""
        ...


class ComparisonRuns(Protocol):
    """Opens session runs in one tracking store and experiment."""

    def session(self, run: NamedRun, tags: Mapping[str, str]) -> AbstractContextManager[SessionRun]:
        """Open a session run; it finishes, or fails, with the block."""
        ...
