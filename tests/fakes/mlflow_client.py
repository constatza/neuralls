"""In-memory stand-in for the ``MlflowClient`` calls model resolution makes.

Runs, registered model versions and aliases live in dicts. A run's checkpoint
files are written under ``artifact_root`` because the artifact bytes are what a
resolution test asserts on (the ``file://`` artifact URI lets the real
``MlflowArtifactLeaseManager`` lease them with no download).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from neuralls.platform.tracking.model_registry import CHECKPOINT_ARTIFACT_PATH_TAG

FAKE_EXPERIMENT_NAME = "fake-experiment"
FAKE_EXPERIMENT_ID = "1"
_RUN_ID_FORMAT = "run-{:04d}"


def _write_artifact(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _FakeRun:
    run_id: str
    run_name: str
    start_time_ms: int
    artifact_dir: Path

    @property
    def info(self) -> SimpleNamespace:
        return SimpleNamespace(
            run_id=self.run_id,
            run_name=self.run_name,
            start_time=self.start_time_ms,
            artifact_uri=self.artifact_dir.as_uri(),
        )

    @property
    def data(self) -> SimpleNamespace:
        return SimpleNamespace(tags={})


@dataclass(frozen=True, slots=True)
class _FakeModelVersion:
    name: str
    version: int
    run_id: str
    tags: Mapping[str, str]


@dataclass(slots=True)
class FakeMlflowClient:
    """Seed with ``add_run`` / ``register_model_version``; pass as the client factory."""

    artifact_root: Path
    tracking_uri: str = "fake://mlflow"
    _runs: list[_FakeRun] = field(default_factory=list)
    _versions: dict[str, list[_FakeModelVersion]] = field(default_factory=dict)
    _aliases: dict[tuple[str, str], int] = field(default_factory=dict)

    def __call__(self, *, tracking_uri: str) -> FakeMlflowClient:
        """Act as the ``MlflowClientFactory``: every call returns this one store."""
        del tracking_uri
        return self

    def add_run(
        self,
        marker: str,
        *,
        run_name: str | None = None,
        start_time_ms: int | None = None,
        checkpoints: Mapping[str, str] | None = None,
    ) -> str:
        """Add a run whose ``model/`` and ``checkpoints/`` hold checkpoints.

        ``checkpoints`` maps relative file name to content and defaults to a
        single ``<marker>.ckpt`` containing ``marker``.
        """
        run_id = _RUN_ID_FORMAT.format(len(self._runs))
        artifact_dir = self.artifact_root / run_id
        files = checkpoints if checkpoints is not None else {f"{marker}.ckpt": marker}
        for directory in ("model", "checkpoints"):
            for name, content in files.items():
                _write_artifact(artifact_dir / directory / name, content)
        self._runs.append(
            _FakeRun(
                run_id=run_id,
                run_name=run_name or marker,
                start_time_ms=len(self._runs) if start_time_ms is None else start_time_ms,
                artifact_dir=artifact_dir,
            )
        )
        return run_id

    def add_run_artifact(self, run_id: str, path: str, text: str) -> None:
        """Add one more artifact file to an existing run."""
        _write_artifact(self._run(run_id).artifact_dir / path, text)

    def register_model_version(
        self,
        name: str,
        run_id: str,
        *,
        pinned_path: str | None,
        alias: str | None = None,
    ) -> int:
        """Register the next version of ``name``; ``None`` pin mimics a legacy version."""
        versions = self._versions.setdefault(name, [])
        version = len(versions) + 1
        tags = {} if pinned_path is None else {CHECKPOINT_ARTIFACT_PATH_TAG: pinned_path}
        versions.append(_FakeModelVersion(name, version, run_id, tags))
        if alias is not None:
            self._aliases[(name, alias)] = version
        return version

    def get_registered_model(self, name: str) -> SimpleNamespace:
        if name not in self._versions:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return SimpleNamespace(name=name)

    def search_model_versions(self, filter_string: str) -> list[_FakeModelVersion]:
        name = filter_string.removeprefix("name='").removesuffix("'")
        return list(self._versions.get(name, []))

    def get_model_version(self, name: str, version: str) -> _FakeModelVersion:
        return self._versions[name][int(version) - 1]

    def get_model_version_by_alias(self, name: str, alias: str) -> _FakeModelVersion:
        return self.get_model_version(name, str(self._aliases[(name, alias)]))

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        if name != FAKE_EXPERIMENT_NAME:
            return None
        return SimpleNamespace(experiment_id=FAKE_EXPERIMENT_ID, name=name)

    def search_experiments(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(experiment_id=FAKE_EXPERIMENT_ID, name=FAKE_EXPERIMENT_NAME)]

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str,
        run_view_type: int,
        max_results: int,
        order_by: list[str],
    ) -> list[_FakeRun]:
        """Newest-first listing; only the unfiltered query the resolver issues is supported."""
        del run_view_type
        if filter_string or order_by != ["attributes.start_time DESC"]:
            raise NotImplementedError(f"FakeMlflowClient cannot evaluate {filter_string!r}")
        if FAKE_EXPERIMENT_ID not in experiment_ids:
            return []
        return sorted(self._runs, key=lambda run: -run.start_time_ms)[:max_results]

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        """Unreachable for ``file://`` artifacts: the lease manager reads them in place."""
        raise AssertionError(f"unexpected download of {path!r} from {run_id}")

    def get_run(self, run_id: str) -> _FakeRun:
        return self._run(run_id)

    def _run(self, run_id: str) -> _FakeRun:
        return next(run for run in self._runs if run.run_id == run_id)
