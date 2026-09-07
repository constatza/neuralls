"""MLflow client utilities for training workflows.

Pure side-effecting helpers that wrap mlflow / MlflowClient calls.
No training logic — kept separate so training.py and training_batch.py
stay free of low-level MLflow plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

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


_COMPARISON_ARTIFACT_DIRS: tuple[str, ...] = ("arrays", "figures", "config")
_COMPARISON_FLAT_FILES: tuple[str, ...] = (
    "comparison.toml",
    "comparison.json",
    "recommendations.json",
    "summary.txt",
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


def log_comparison_artifacts_to_mlflow(
    tracking_uri: str,
    run_id: str,
    work_root: Path,
) -> None:
    """Upload comparison artifacts to an existing MLflow run.

    Uploads only the structured comparison outputs, excluding downloaded model
    checkpoints that may reside in the same working directory.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run to upload into.
        work_root: Working directory produced by the comparison run.
    """
    _log_artifact_paths(
        tracking_uri=tracking_uri,
        run_id=run_id,
        root=work_root,
        dirs=_COMPARISON_ARTIFACT_DIRS,
        flat_files=_COMPARISON_FLAT_FILES,
        artifact_path_per_dir=True,
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


def _search_finished_runs(
    *,
    client: MlflowClient,
    mlflow_experiment_name: str,
    tag_filters: Mapping[str, str],
    max_results: int,
) -> list[Run]:
    """Search FINISHED runs in one experiment matching every given tag, newest first.

    Shared by every "has this already run" reuse check — each caller owns its
    own tag set and its own policy for picking a match among the results;
    this only owns the experiment lookup and filter-string mechanics.

    Returns:
        Matching runs, or an empty list when the experiment doesn't exist —
        callers don't need a separate "experiment missing" branch.
    """
    experiment = client.get_experiment_by_name(mlflow_experiment_name)
    if experiment is None:
        return []
    clauses = ["attributes.status = 'FINISHED'"] + [
        f"tags.{key} = '{value}'" for key, value in tag_filters.items()
    ]
    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(clauses),
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )


def find_successful_run(
    *,
    tracking_uri: str,
    mlflow_experiment_name: str,
    assignment_id: str,
    dataset_hash: str | None = None,
) -> str | None:
    """Return the run_id of the most recent FINISHED run with a checkpoint artifact.

    A FINISHED status alone isn't sufficient: dlkit closes the MLflow run (status=FINISHED)
    before neuralls's own post-training checkpoint-upload step runs, so a run can be
    FINISHED with no checkpoint if that later step fails. This walks FINISHED runs
    newest-first and returns the first one dlkit's has_checkpoint_artifact() confirms
    actually has a checkpoint, so a broken newest attempt doesn't permanently shadow an
    older, genuinely complete run for the same assignment.

    Used to decide whether an assignment (one job run on one dataset) has already
    completed successfully, so a batch rerun can skip retraining it, and by eval to pick
    which run to evaluate.

    Args:
        tracking_uri: MLflow tracking URI.
        mlflow_experiment_name: MLflow experiment (bucket) to search in.
        assignment_id: The assignment's stable id (tagged on its training run).
        dataset_hash: When given, also require the candidate run's tagged
            ``dataset_hash`` to match — so a dataset regenerated since that run
            no longer reads as "already trained" and a rerun picks it up.

    Returns:
        The matching run's run_id, or None if no FINISHED run with a checkpoint exists yet.
    """
    from dlkit.mlflow import has_checkpoint_artifact
    from mlflow.tracking import MlflowClient

    tag_filters = {"assignment_id": assignment_id}
    if dataset_hash is not None:
        tag_filters["dataset_hash"] = dataset_hash

    runs = _search_finished_runs(
        client=MlflowClient(tracking_uri=tracking_uri),
        mlflow_experiment_name=mlflow_experiment_name,
        tag_filters=tag_filters,
        max_results=20,
    )
    for run in runs:
        if has_checkpoint_artifact(run.info.run_id, tracking_uri=tracking_uri):
            return run.info.run_id
        logger.warning(
            "Run {} for assignment '{}' is FINISHED but has no checkpoint artifact; "
            "skipping to an older candidate if one exists.",
            run.info.run_id,
            assignment_id,
        )
    return None


def find_successful_comparison_run(
    *,
    tracking_uri: str,
    mlflow_experiment_name: str,
    comparison_id: str,
    checkpoint_dependency_hash: str,
) -> str | None:
    """Return the run_id of the most recent FINISHED comparison run with this dependency hash.

    Used to decide whether a ``[[comparisons]]`` entry has already been run against
    the exact same resolved preconditioner checkpoints, so a batch rerun can skip
    recomputing it. Unlike `find_successful_run`, no separate artifact check is
    needed: a comparison run only reaches FINISHED after its result artifacts are
    staged and logged inside the same ``mlflow.start_run()`` block that created it
    (no split closing step like dlkit's training runs have).

    Args:
        tracking_uri: MLflow tracking URI.
        mlflow_experiment_name: MLflow experiment (bucket) to search in.
        comparison_id: The comparison entry's stable id (tagged on its run).
        checkpoint_dependency_hash: Hash of the resolved checkpoints this
            comparison depends on — a fresh training run changes this and
            invalidates the cached comparison.

    Returns:
        The matching run's run_id, or None if no matching FINISHED run exists yet.
    """
    from mlflow.tracking import MlflowClient

    runs = _search_finished_runs(
        client=MlflowClient(tracking_uri=tracking_uri),
        mlflow_experiment_name=mlflow_experiment_name,
        tag_filters={
            "comparison_id": comparison_id,
            "checkpoint_dependency_hash": checkpoint_dependency_hash,
        },
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def log_diagnostics_to_mlflow(
    tracking_uri: str,
    run_id: str,
    diagnostics: object,
    figure_path: Path,
) -> None:
    """Log diagnostics metrics and figure to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        diagnostics: Diagnostics result with ``metrics`` dict attribute.
        figure_path: Path to the figure file to upload.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    metrics = getattr(diagnostics, "metrics", {})
    for key, value in metrics.items():
        client.log_metric(run_id, key, float(value))
    if figure_path.exists():
        client.log_artifact(run_id, str(figure_path), artifact_path="figures")
