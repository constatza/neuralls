"""Typed MLflow helpers for resolving paths and lightweight logging."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

type SessionRunStatus = Literal["FINISHED", "FAILED"]

import mlflow
from dlkit.infrastructure.io import url_resolver
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient

from neuralls.platform.config.loaders import load_tracking_config
from neuralls.platform.config.resolution import (
    MlflowPaths,
    build_mlflow_environment,
    build_sqlite_tracking_uri,
    resolve_mlflow_paths,
    to_mlflow_artifact_location,
)
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT


@dataclass(frozen=True)
class MlflowRunConfig:
    """Declarative MLflow run settings."""

    experiment_name: str
    run_name: str
    tags: Mapping[str, str]
    paths: MlflowPaths
    workspace_root: Path


@dataclass(frozen=True)
class MlflowRunState:
    """Active MLflow run handle."""

    run: ActiveRun
    started: bool


@dataclass(frozen=True)
class MlflowRuntimeEnvironment:
    """Resolved MLflow environment values for one workflow.

    Attributes:
        env: Environment variables to export for the workflow process.
        tracking_uri: Resolved MLflow tracking URI.
        artifact_uri: Resolved MLflow artifact URI, if any.
        is_explicit: Whether ``tracking_uri`` came from an explicit source
            (the ``MLFLOW_TRACKING_URI`` env var, a case config, or
            ``configs/tracking.toml``) rather than the local sqlite fallback
            derived from an output directory.
    """

    env: dict[str, str]
    tracking_uri: str
    artifact_uri: str | None
    is_explicit: bool


DEFAULT_ARTIFACT_SUBDIRS: tuple[str, ...] = (
    "checkpoints",
    "figures",
    "predictions",
    "reports",
    "metrics",
)
_MLFLOW_METRIC_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_REMOTE_TRACKING_SCHEMES = frozenset({"http", "https"})
_LOCAL_ARTIFACT_SCHEME = "file"


def build_run_config(
    *,
    settings: Any,
    workspace_root: Path,
    dataset_id: str,
    model_name: str,
    session_name: str | None = None,
    enabled: bool = True,
) -> MlflowRunConfig | None:
    """Create a run config when MLflow is enabled."""
    tracking_cfg = getattr(settings, "tracking", None)
    backend = getattr(tracking_cfg, "backend", "none") if tracking_cfg is not None else "none"
    if not enabled or backend != "mlflow":
        return None
    output_dir = workspace_root
    default_tracking_uri = None
    default_artifact_location = None
    if output_dir:
        output_root = Path(output_dir).resolve()
        default_tracking_uri = build_sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
        default_artifact_location = str((output_root / "mlartifacts").resolve())
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
    paths = resolve_mlflow_paths(
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        project_root=DEFAULT_PROJECT_ROOT,
        workspace_root=workspace_root,
        default_tracking_uri=default_tracking_uri,
        default_artifact_location=default_artifact_location,
    )
    experiment_name = dataset_id
    run_name = model_name
    tags = make_run_tags(dataset_id, model_name, session_name)
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
        paths=paths,
        workspace_root=workspace_root,
    )


def make_run_tags(
    dataset_id: str,
    model_name: str,
    session_name: str | None = None,
) -> dict[str, str]:
    """Build minimal tag set for run context."""
    tags = {"dataset": dataset_id, "model": model_name}
    if session_name:
        tags["session"] = session_name
    return tags


def sanitize_metric_key_segment(name: str, default: str = "unnamed") -> str:
    """Return an MLflow-safe metric key segment."""
    sanitized = _MLFLOW_METRIC_SEGMENT_PATTERN.sub("_", name).strip("._-")
    return sanitized or default


def quote_filter_value(value: str) -> str:
    """Escape one string value for MLflow search filters."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def runtime_paths_from_env(runtime_mlflow_env: Mapping[str, str]) -> MlflowPaths:
    """Build resolved MLflow paths from an active MLflow environment."""
    return MlflowPaths(
        tracking_uri=runtime_mlflow_env["MLFLOW_TRACKING_URI"],
        artifact_uri=runtime_mlflow_env.get("MLFLOW_ARTIFACT_URI"),
    )


def resolve_runtime_tracking_config() -> tuple[str | None, str]:
    """Resolve tracking URI and artifact destination from runtime env."""
    return os.environ.get("MLFLOW_TRACKING_URI"), os.environ.get("MLFLOW_ARTIFACT_URI", "")


def build_runtime_environment(
    output_root: Path | str | None,
    *,
    default_output_root: Path,
) -> MlflowRuntimeEnvironment:
    """Resolve the MLflow environment for a runtime workflow execution.

    Precedence:
    1. ``MLFLOW_TRACKING_URI`` env var (already set in the process)
    2. ``configs/tracking.toml`` (shared project-level source of truth)
    3. Local sqlite fallback under *output_root* / *default_output_root*
    """
    existing_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if existing_uri:
        existing_artifact_uri = os.environ.get("MLFLOW_ARTIFACT_URI")
        env = {"MLFLOW_TRACKING_URI": existing_uri}
        if existing_artifact_uri:
            env["MLFLOW_ARTIFACT_URI"] = existing_artifact_uri
        return MlflowRuntimeEnvironment(
            env=env,
            tracking_uri=existing_uri,
            artifact_uri=existing_artifact_uri,
            is_explicit=True,
        )

    shared = load_tracking_config()
    if shared is not None and shared.uri is not None:
        env = build_mlflow_environment(
            tracking_uri=shared.uri,
            artifacts_destination=shared.artifacts_destination,
        )
        return MlflowRuntimeEnvironment(
            env=env,
            tracking_uri=env["MLFLOW_TRACKING_URI"],
            artifact_uri=env.get("MLFLOW_ARTIFACT_URI"),
            is_explicit=True,
        )

    permanent_root = Path(output_root).resolve() if output_root else default_output_root
    env = build_mlflow_environment(
        tracking_uri=build_sqlite_tracking_uri(permanent_root / "mlruns" / "mlflow.db"),
        artifacts_destination=str((permanent_root / "mlartifacts").resolve()),
    )
    return MlflowRuntimeEnvironment(
        env=env,
        tracking_uri=env["MLFLOW_TRACKING_URI"],
        artifact_uri=env.get("MLFLOW_ARTIFACT_URI"),
        is_explicit=False,
    )


def build_workflow_environment(
    *,
    tracking_uri: str | None,
    artifact_location: str | None,
    default_output_root: Path,
    config_path: Path | None = None,
) -> MlflowRuntimeEnvironment:
    """Build the MLflow environment for a case-driven workflow.

    Precedence when *tracking_uri* is ``None``:
    1. ``configs/tracking.toml`` (shared project-level source of truth)
    2. Local sqlite fallback under *default_output_root*
    """
    resolved_uri = tracking_uri
    resolved_artifacts = artifact_location
    if resolved_uri is None:
        shared = load_tracking_config()
        if shared is not None:
            resolved_uri = shared.uri
            resolved_artifacts = resolved_artifacts or shared.artifacts_destination
    is_explicit = resolved_uri is not None
    if resolved_uri is None:
        env = build_mlflow_environment(
            tracking_uri=build_sqlite_tracking_uri(default_output_root / "mlruns" / "mlflow.db"),
            artifacts_destination=str((default_output_root / "mlartifacts").resolve()),
        )
    else:
        env = build_mlflow_environment(
            tracking_uri=resolved_uri,
            artifacts_destination=resolved_artifacts,
            config_path=config_path,
        )
    return MlflowRuntimeEnvironment(
        env=env,
        tracking_uri=env["MLFLOW_TRACKING_URI"],
        artifact_uri=env.get("MLFLOW_ARTIFACT_URI"),
        is_explicit=is_explicit,
    )


def _assert_remote_artifact_location(
    *,
    tracking_uri: str,
    experiment_name: str,
    artifact_location: str | None,
) -> None:
    """Fail loudly if a remote-tracked experiment's artifacts resolve to a local path.

    An experiment's ``artifact_location`` is fixed at creation time and MLflow
    provides no API to change it afterward. If an experiment was ever created
    with a local ``file://`` location (e.g. by the CWD-fallback bug this
    guards against), every future run into it keeps writing local files no
    matter how correctly tracking is configured — silently. Catch that here,
    before any run/upload happens, instead of letting it recreate hashed
    local directories.
    """
    if url_resolver.scheme(tracking_uri) not in _REMOTE_TRACKING_SCHEMES:
        return
    if (
        artifact_location is None
        or url_resolver.scheme(artifact_location) != _LOCAL_ARTIFACT_SCHEME
    ):
        return
    raise RuntimeError(
        f"MLflow experiment {experiment_name!r} has a local artifact_location "
        f"({artifact_location!r}) despite using remote tracking at {tracking_uri!r}. "
        "This is permanent — MLflow experiments' artifact_location cannot be "
        "changed after creation. Delete and recreate the experiment on the "
        "server (mlflow.delete_experiment + a fresh mlflow.create_experiment "
        "with no local artifact_location), or rename it in the case config's "
        "[names] section to start a fresh, correctly-configured experiment."
    )


def _optional_artifact_location(artifact_uri: str | None) -> str | None:
    """Convert an optional local artifact root into an MLflow-compatible location."""
    if artifact_uri is None:
        return None
    return to_mlflow_artifact_location(artifact_uri)


def ensure_experiment(name: str, paths: MlflowPaths) -> str:
    """Create or reuse an experiment and return its id.

    When no artifact destination was explicitly configured, omits
    ``artifact_location`` entirely rather than defaulting it to the client's
    CWD — the tracking server picks its own default artifact root instead.
    """
    mlflow.set_tracking_uri(paths.tracking_uri)
    existing = mlflow.get_experiment_by_name(name)
    if existing:
        _assert_remote_artifact_location(
            tracking_uri=paths.tracking_uri,
            experiment_name=name,
            artifact_location=existing.artifact_location,
        )
        return existing.experiment_id
    experiment_id = mlflow.create_experiment(
        name=name,
        artifact_location=_optional_artifact_location(paths.artifact_uri),
    )
    created = mlflow.get_experiment(experiment_id)
    _assert_remote_artifact_location(
        tracking_uri=paths.tracking_uri,
        experiment_name=name,
        artifact_location=created.artifact_location,
    )
    return experiment_id


def create_session_parent_run(
    *,
    tracking_uri: str,
    artifact_uri: str | None,
    run_name: str,
    tags: Mapping[str, str],
    experiment_name: str,
) -> str:
    """Create a non-active MLflow parent run wrapping one batch session.

    Uses a raw ``MlflowClient.create_run`` rather than ``mlflow.start_run`` so the
    parent never becomes the process's active run — callers that hand off to
    out-of-process/independent run creators (e.g. dlkit) rely on the parent
    staying inactive so their own run creation doesn't conflict with it.

    Args:
        tracking_uri: MLflow tracking URI.
        artifact_uri: Optional artifact location for a newly created experiment.
        run_name: Display name for the session parent run.
        tags: MLflow tags to attach to the parent run.
        experiment_name: Experiment to create the run under.

    Returns:
        The new parent run's UUID.
    """
    experiment_id = ensure_experiment(
        experiment_name,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_uri),
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    parent_run = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.runName": run_name, **tags},
    )
    return parent_run.info.run_id


def finalize_session_parent_run(
    *, tracking_uri: str, run_id: str, status: SessionRunStatus
) -> None:
    """Terminate a session parent run created via ``create_session_parent_run``."""
    MlflowClient(tracking_uri=tracking_uri).set_terminated(run_id, status=status)


def start_run_if_needed(
    exp_id: str,
    run_name: str,
    tags: Mapping[str, str] | None = None,
) -> tuple[ActiveRun, bool]:
    """Reuse active run or start a new one with tags."""
    active = mlflow.active_run()
    if active:
        return active, False
    run = mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=dict(tags or {}))
    return run, True


def open_run(config: MlflowRunConfig) -> MlflowRunState:
    """Start or reuse a run and return its state."""
    exp_id = ensure_experiment(config.experiment_name, config.paths)
    run, started = start_run_if_needed(exp_id, config.run_name, config.tags)
    return MlflowRunState(run=run, started=started)


def collect_artifacts(workspace: Path, allowlist: Sequence[str]) -> list[Path]:
    """Return existing artifact paths under workspace using an allowlist."""
    root = Path(workspace)
    return [path for name in allowlist if (path := root / name).exists()]


def log_artifacts(run: ActiveRun, artifacts: Sequence[Path]) -> None:
    """Log directories/files to MLflow for the given run."""
    if not artifacts:
        return
    for path in artifacts:
        mlflow.log_artifacts(str(path), run_id=run.info.run_id)


def log_metrics(run: ActiveRun, metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow for the given run."""
    if not metrics:
        return
    mlflow.log_metrics(dict(metrics), step=step, run_id=run.info.run_id)


def finalize_run(
    state: MlflowRunState | None,
    *,
    metrics: Mapping[str, float] | None = None,
    workspace_root: Path,
    allowlist: Sequence[str] = DEFAULT_ARTIFACT_SUBDIRS,
    failed: bool = False,
) -> None:
    """Upload artifacts/metrics and end run if we started it."""
    if state is None:
        return
    artifacts = collect_artifacts(workspace_root, allowlist)
    log_artifacts(state.run, artifacts)
    if metrics:
        log_metrics(state.run, metrics)
    if state.started:
        mlflow.end_run(status="FAILED" if failed else "FINISHED")
