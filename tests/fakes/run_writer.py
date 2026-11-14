"""In-memory ``RunWriter`` keeping every run a tracking adapter creates and what it writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from neuralls.platform.tracking.run_writer import RunStatus


@dataclass(slots=True)
class RecordedRun:
    """One run and everything written into it; ``artifacts`` holds file bytes at log time."""

    experiment_id: str
    name: str
    tags: dict[str, str]
    parent_run_id: str | None
    status: str = "RUNNING"
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[tuple[str | None, str], bytes] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    dicts: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(slots=True)
class RecordingRunWriter:
    """Runs are named ``run-1``, ``run-2``, ... in creation order."""

    runs: dict[str, RecordedRun] = field(default_factory=dict)

    def create_run(
        self,
        experiment_id: str,
        name: str,
        tags: Mapping[str, str],
        *,
        parent_run_id: str | None = None,
    ) -> str:
        run_id = f"run-{len(self.runs) + 1}"
        self.runs[run_id] = RecordedRun(experiment_id, name, dict(tags), parent_run_id)
        return run_id

    def finish(self, run_id: str, status: RunStatus) -> None:
        self._run(run_id).status = status

    def log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        self._run(run_id).params.update(params)

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        self._run(run_id).metrics.update(metrics)

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        self._run(run_id).tags.update(tags)

    def log_artifact(self, run_id: str, local_path: Path, directory: str | None) -> None:
        self._run(run_id).artifacts[directory, local_path.name] = local_path.read_bytes()

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self._run(run_id).texts[artifact_file] = text

    def log_dict(self, run_id: str, data: Mapping[str, object], artifact_file: str) -> None:
        self._run(run_id).dicts[artifact_file] = dict(data)

    def _run(self, run_id: str) -> RecordedRun:
        return self.runs.setdefault(run_id, RecordedRun("", run_id, {}, None))
