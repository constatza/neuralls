"""Real local MLflow store fixtures: an experiment with file-backed artifacts and logged runs.

Registered in ``tests/conftest.py`` so the IO-boundary directories that hold real-store tests
(``tests/integration``, ``tests/platform``) share one definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import pytest
from dlkit.infrastructure.io import url_resolver
from mlflow.tracking import MlflowClient


def _artifact_location(path: Path) -> str:
    """Convert a local artifacts root into the file:// URI form MLflow expects.

    Args:
        path: Local directory to use as an experiment artifact root.

    Returns:
        A ``file://`` URI string.
    """
    return url_resolver.build_uri(path.resolve(), scheme="file")


class LoggedRunFactory(Protocol):
    """Callable that logs one MLflow run with a checkpoint marker artifact."""

    def __call__(
        self,
        marker: str,
        *,
        run_name: str | None = None,
        start_time_ms: int | None = None,
    ) -> str: ...


@pytest.fixture
def mlflow_experiment(tmp_path: Path, mlflow_tracking_uri: str) -> str:
    """Create and activate a real local MLflow experiment with file-backed artifacts.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI to bind the experiment to.

    Returns:
        The created experiment's name.
    """
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = f"workflow-test-{tmp_path.name}"
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=_artifact_location(artifacts_dir),
        )
    return experiment_name


@pytest.fixture
def log_run_with_checkpoint(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
) -> LoggedRunFactory:
    """Factory fixture: log one real run with checkpoint artifacts.

    Each call logs a distinct run under ``mlflow_experiment`` containing a
    ``.ckpt`` artifact under both the ``model`` and ``checkpoints`` artifact
    paths whose content equals ``marker``, so callers can assert exactly which
    run's checkpoint was resolved. Logging under ``checkpoints`` lets the same
    fixture back both raw-run-reference resolution (``model`` fallback) and
    registration (which requires a ``checkpoints/`` artifact to pin).
    ``start_time_ms`` optionally overrides the run's start time to make
    ordering-sensitive assertions (e.g. "latest run") deterministic instead of
    relying on real wall-clock ordering.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI.
        mlflow_experiment: Active experiment name (ensures setup ordering).

    Returns:
        A callable ``(marker, *, run_name=None, start_time_ms=None) -> run_id``.
    """
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        raise RuntimeError(f"Experiment '{mlflow_experiment}' was not found after creation.")

    def _log_run(
        marker: str,
        *,
        run_name: str | None = None,
        start_time_ms: int | None = None,
    ) -> str:
        checkpoint_file = tmp_path / f"{marker}.ckpt"
        checkpoint_file.write_text(marker, encoding="utf-8")
        run_id = client.create_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name or marker,
            start_time=start_time_ms,
        ).info.run_id
        client.log_artifact(run_id, str(checkpoint_file), artifact_path="model")
        client.log_artifact(run_id, str(checkpoint_file), artifact_path="checkpoints")
        client.set_terminated(run_id)
        return run_id

    return _log_run


class LoggedNamedCheckpointsRunFactory(Protocol):
    """Callable that logs one MLflow run with several named checkpoint files."""

    def __call__(self, checkpoints: Mapping[str, str]) -> str: ...


@pytest.fixture
def log_run_with_named_checkpoints(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
) -> LoggedNamedCheckpointsRunFactory:
    """Factory fixture: log one real run with several named checkpoint files.

    Each call logs a distinct run under ``mlflow_experiment`` with one
    ``checkpoints/<name>`` artifact per ``(name, content)`` pair given —
    e.g. ``{"best.ckpt": "best", "last.ckpt": "last"}`` — so callers can
    exercise ambiguous-checkpoint selection and registration pinning against
    multiple real candidate files.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI.
        mlflow_experiment: Active experiment name (ensures setup ordering).

    Returns:
        A callable ``(checkpoints: Mapping[str, str]) -> run_id``.
    """

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        raise RuntimeError(f"Experiment '{mlflow_experiment}' was not found after creation.")

    def _log_run(checkpoints: Mapping[str, str]) -> str:
        run_id = client.create_run(experiment_id=experiment.experiment_id).info.run_id
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, content in checkpoints.items():
            checkpoint_file = run_dir / name
            checkpoint_file.write_text(content, encoding="utf-8")
            client.log_artifact(run_id, str(checkpoint_file), artifact_path="checkpoints")
        client.set_terminated(run_id)
        return run_id

    return _log_run
