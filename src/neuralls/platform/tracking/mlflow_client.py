"""MLflow client utilities for training workflows.

Pure side-effecting helpers that wrap mlflow / MlflowClient calls.
No training logic — kept separate so training.py and training_batch.py
stay free of low-level MLflow plumbing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from loguru import logger
from mlflow.tracking import MlflowClient

# Called as ``factory(tracking_uri=...)``; tests substitute a fake client here.
type MlflowClientFactory = Callable[..., MlflowClient]

_WORKSPACE_ARTIFACT_DIRS: tuple[str, ...] = (
    "config",
    "figures",
    "metrics",
    "predictions",
)


def fetch_mlflow_metrics(run_id: str, tracking_uri: str) -> dict[str, float]:
    """Query MLflow for all metrics belonging to a run.

    Args:
        run_id: MLflow run UUID.
        tracking_uri: SQLite or HTTP tracking URI.

    Returns:
        Flat dict of ``{metric_key: value}`` for the run.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def log_metric_to_run(run_id: str, key: str, value: float, tracking_uri: str) -> None:
    """Log a single metric onto an existing (possibly already-closed) MLflow run.

    For callers outside an active ``mlflow.start_run()`` context — e.g. a
    multirun sweep's lifecycle hooks, fired after a child's own run has
    already closed — that only have the child's ``run_id`` on hand.

    Args:
        run_id: MLflow run UUID to log onto.
        key: Metric name.
        value: Metric value.
        tracking_uri: SQLite or HTTP tracking URI.
    """
    from mlflow.tracking import MlflowClient

    MlflowClient(tracking_uri=tracking_uri).log_metric(run_id, key, float(value))


def _log_artifact_paths(
    *,
    tracking_uri: str,
    run_id: str,
    root: Path,
    dirs: Sequence[str],
    flat_files: Sequence[str] = (),
    artifact_path_per_dir: bool = False,
) -> None:
    """Upload existing directories/files under root to an MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run to upload into.
        root: Local directory the paths are resolved relative to.
        dirs: Subdirectory names to upload wholesale, when present.
        flat_files: Individual file names to upload, when present.
        artifact_path_per_dir: When True, nest each directory's artifacts under
            an artifact_path matching its own name; when False, upload flat.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    for name in dirs:
        path = root / name
        if not path.exists():
            continue
        artifact_path = name if artifact_path_per_dir else None
        client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
    for name in flat_files:
        path = root / name
        if not path.exists():
            continue
        client.log_artifact(run_id, str(path))


def log_artifacts_to_mlflow(
    tracking_uri: str,
    run_id: str,
    workspace_root: Path,
) -> None:
    """Upload the staged workspace artifacts to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run to upload into.
        workspace_root: Local directory whose contents are uploaded.
    """
    _log_artifact_paths(
        tracking_uri=tracking_uri,
        run_id=run_id,
        root=workspace_root,
        dirs=_WORKSPACE_ARTIFACT_DIRS,
    )


def log_batch_artifacts_to_mlflow(
    tracking_uri: str,
    run_id: str,
    work_root: Path,
    flat_files: Sequence[str],
) -> None:
    """Upload batch summary artifacts (barplot + label map) to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run to upload into (the batch's session parent run).
        work_root: Local scratch directory the summary artifacts were staged into.
        flat_files: Individual file names under ``work_root`` to upload, when present.
    """
    _log_artifact_paths(
        tracking_uri=tracking_uri,
        run_id=run_id,
        root=work_root,
        dirs=(),
        flat_files=flat_files,
    )


def mark_run_failed(*, run_id: str, tracking_uri: str) -> None:
    """Best-effort mark a run FAILED so it never reads as successful when it isn't.

    Args:
        run_id: Run to mark.
        tracking_uri: MLflow tracking URI.
    """
    from mlflow.tracking import MlflowClient

    try:
        MlflowClient(tracking_uri=tracking_uri).set_terminated(run_id, status="FAILED")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not mark run {} as FAILED: {}", run_id, exc)


def find_mlflow_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
) -> tuple[str, str] | None:
    """Look up the most recent training run by experiment + run name.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Experiment name to search in.
        run_name: Optional run name filter.

    Returns:
        ``(experiment_id, run_id)`` when found, otherwise ``None``.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        return None
    if run_name is None:
        return experiment.experiment_id, runs[0].info.run_id

    for run in runs:
        candidate_name = getattr(run.info, "run_name", None) or run.data.tags.get("mlflow.runName")
        if candidate_name == run_name:
            return experiment.experiment_id, run.info.run_id
    return None


def log_diagnostics_to_mlflow(
    tracking_uri: str,
    run_id: str,
    diagnostics: object,
) -> None:
    """Log diagnostics metrics to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        diagnostics: Diagnostics result with ``metrics`` dict attribute.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    metrics = getattr(diagnostics, "metrics", {})
    for key, value in metrics.items():
        client.log_metric(run_id, key, float(value))
