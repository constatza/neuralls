"""Writes to MLflow runs named by id, through a client bound to one tracking store.

This is the only way the package writes tracking data. MLflow's fluent API (``mlflow.log_*``,
``mlflow.start_run``) reads the process-wide active run and tracking URI instead; with neither
set it silently starts a run on ``./mlflow.db``, outside every store a case configures. A
writer names its store when it is built and its run on every call, so a missing run is a
missing argument, never a new store. ``tests/architecture/test_io_boundary.py`` keeps the
fluent API out of ``src/``.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags

PARENT_RUN_TAG = "mlflow.parentRunId"
"""The tag MLflow nests a child run under its parent by."""

RUN_DESCRIPTION_TAG = "mlflow.note.content"
"""The reserved tag MLflow shows as a run's description."""

type RunStatus = Literal["FINISHED", "FAILED"]
"""How a run ends: its block completed, or raised."""


class RunWriter(Protocol):
    """Creates, finishes and writes into tracking runs named by id."""

    def create_run(
        self,
        experiment_id: str,
        name: str,
        tags: Mapping[str, str],
        *,
        parent_run_id: str | None = None,
    ) -> str:
        """Create a running run and return its id."""
        ...

    def finish(self, run_id: str, status: RunStatus) -> None:
        """Terminate a run with ``status``."""
        ...

    def log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        """Log string parameters onto a run."""
        ...

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        """Log scalar metrics onto a run."""
        ...

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        """Set tags on a run."""
        ...

    def log_artifact(self, run_id: str, local_path: Path, directory: str | None) -> None:
        """Upload a local file into ``directory`` of the run's artifacts."""
        ...

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        """Store ``text`` as an artifact file of the run."""
        ...

    def log_dict(self, run_id: str, data: Mapping[str, object], artifact_file: str) -> None:
        """Store ``data`` as a JSON artifact file of the run."""
        ...


@dataclass(frozen=True, slots=True)
class MlflowRunWriter:
    """Creates, finishes and writes into runs of the store ``client`` is bound to."""

    client: MlflowClient

    def create_run(
        self,
        experiment_id: str,
        name: str,
        tags: Mapping[str, str],
        *,
        parent_run_id: str | None = None,
    ) -> str:
        """Create a running run and return its id.

        The run carries the same source and user tags ``mlflow.start_run`` would add, and a
        child names its parent the way MLflow's UI nests runs.
        """
        parent = {} if parent_run_id is None else {PARENT_RUN_TAG: parent_run_id}
        run = self.client.create_run(
            experiment_id, tags=resolve_tags({**tags, **parent}), run_name=name
        )
        return run.info.run_id

    def finish(self, run_id: str, status: RunStatus) -> None:
        """Terminate a run with ``status``."""
        self.client.set_terminated(run_id, status=status)

    def log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        """Log string parameters onto a run."""
        if params:
            self.client.log_batch(run_id, params=[Param(k, v) for k, v in params.items()])

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        """Log scalar metrics onto a run, all with one timestamp."""
        if not metrics:
            return
        now = int(time.time() * 1000)
        self.client.log_batch(
            run_id, metrics=[Metric(k, float(v), now, 0) for k, v in metrics.items()]
        )

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        """Set tags on a run."""
        if tags:
            self.client.log_batch(run_id, tags=[RunTag(k, v) for k, v in tags.items()])

    def log_artifact(self, run_id: str, local_path: Path, directory: str | None) -> None:
        """Upload a local file into ``directory`` of the run's artifacts (its root for ``None``)."""
        self.client.log_artifact(run_id, str(local_path), artifact_path=directory)

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        """Store ``text`` as an artifact file of the run, with LF line endings.

        ``MlflowClient.log_text`` writes in text mode, so on Windows it stores CRLF and the
        artifact's bytes differ from the same payload logged as bytes on any other platform.
        """
        artifact = Path(artifact_file)
        with tempfile.TemporaryDirectory() as staging:
            path = Path(staging) / artifact.name
            path.write_text(text, encoding="utf-8", newline="\n")
            parent = artifact.parent.as_posix()
            self.client.log_artifact(run_id, str(path), None if parent == "." else parent)

    def log_dict(self, run_id: str, data: Mapping[str, object], artifact_file: str) -> None:
        """Store ``data`` as a JSON artifact file of the run."""
        self.client.log_dict(run_id, dict(data), artifact_file)


@contextmanager
def open_run(
    writer: RunWriter,
    experiment_id: str,
    name: str,
    tags: Mapping[str, str],
    *,
    parent_run_id: str | None = None,
) -> Generator[str]:
    """Create a run, yield its id, and finish it, as failed if the block raised."""
    run_id = writer.create_run(experiment_id, name, tags, parent_run_id=parent_run_id)
    try:
        yield run_id
    except BaseException:
        writer.finish(run_id, "FAILED")
        raise
    writer.finish(run_id, "FINISHED")
