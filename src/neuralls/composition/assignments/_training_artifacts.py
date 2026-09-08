"""MLflow run setup, artifact staging, and checkpoint resolution for the training workflow."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.assembler import AssignmentIdentity
from neuralls.composition.assignments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.composition.tracking.run_specs import build_training_run_spec, format_run_timestamp
from neuralls.platform.config.models.experiments import AssignmentEntry, ExperimentNamesConfig
from neuralls.platform.reporting.training_diagnostics import (
    compute_diagnostics,
    write_diagnostics_figure,
)
from neuralls.platform.storage.checkpoints import get_latest_checkpoint
from neuralls.platform.storage.training_artifacts import (
    coerce_jsonable,
    save_training_predictions,
)
from neuralls.platform.tracking.artifact_access import ArtifactLeaseManager
from neuralls.platform.tracking.artifact_selection import CHECKPOINT_ARTIFACT_DIR
from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint
from neuralls.platform.tracking.mlflow import MlflowRunConfig, runtime_paths_from_env
from neuralls.platform.tracking.mlflow_client import (
    find_mlflow_run,
    log_diagnostics_to_mlflow,
)
from neuralls.platform.tracking.model_registry import upload_checkpoint_artifacts

BEST_CHECKPOINT_ARTIFACT_KEY = "best_checkpoint"
LAST_CHECKPOINT_ARTIFACT_KEY = "last_checkpoint"
RETAINED_CHECKPOINTS_DIR_NAME = "retained-checkpoints"


@dataclass(frozen=True)
class MlflowCoordinates:
    """The three values that together address one MLflow run.

    Attributes:
        tracking_uri: MLflow tracking URI the run lives under.
        experiment_id: MLflow experiment id owning the run.
        run_id: MLflow run id.
    """

    tracking_uri: str
    experiment_id: str
    run_id: str


def _resolve_training_experiment_name(mlflow_experiment_name: str | None) -> str:
    """Resolve the training MLflow experiment name from caller input or config defaults.

    Args:
        mlflow_experiment_name: Caller-supplied name, or None to use the config default.

    Returns:
        Resolved experiment name.
    """
    if mlflow_experiment_name is not None:
        return mlflow_experiment_name
    return ExperimentNamesConfig().training


def _build_training_run_config(
    *,
    identity: AssignmentIdentity,
    mlflow_experiment_name: str | None,
    runtime_mlflow_env: Mapping[str, str],
    workspace_root: Path,
    include_timestamp: bool = True,
) -> MlflowRunConfig:
    """Build the execute()-time MLflow run config for training.

    Args:
        identity: Resolved assignment identity. A run is tagged with the
            structured training tags only when its assignment, dataset, and job
            registry ids are all present; otherwise it falls back to a bare,
            untagged run named after the assignment's display name.
        mlflow_experiment_name: Override for the MLflow experiment bucket name.
        runtime_mlflow_env: MLflow environment variable mapping.
        workspace_root: Root directory for the training workspace.
        include_timestamp: Whether to append a timestamp to the run name.
            False for batch runs, where the parent/sweep already disambiguates
            children without one.

    Returns:
        Fully configured MlflowRunConfig.
    """
    experiment_name = _resolve_training_experiment_name(mlflow_experiment_name)
    paths = runtime_paths_from_env(runtime_mlflow_env)
    display_name = identity.assignment_display_name
    if identity.assignment_id and identity.dataset_registry_id and identity.job_registry_id:
        entry = AssignmentEntry(
            id=identity.assignment_id,
            dataset=identity.dataset_registry_id,
            job=identity.job_registry_id,
            display_name=display_name,
        )
        return build_training_run_spec(
            entry=entry,
            experiment_name=experiment_name,
            paths=paths,
            workspace_root=workspace_root,
            include_timestamp=include_timestamp,
        )
    ts = f" | {format_run_timestamp()}" if include_timestamp else ""
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=f"{display_name}{ts}",
        tags={},
        paths=paths,
        workspace_root=workspace_root,
    )


def _resolve_mlflow_run_ids(
    *,
    training_result: Any,
    fallback_tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
) -> MlflowCoordinates | None:
    """Resolve MLflow tracking URI, experiment ID, and run ID for a training run.

    Args:
        training_result: DLKit training result object.
        fallback_tracking_uri: Tracking URI to use when not found in result metrics.
        experiment_name: MLflow experiment name for fallback lookup.
        run_name: MLflow run name for fallback lookup.

    Returns:
        The run's `MlflowCoordinates`, or None if unresolvable.
    """
    metrics = getattr(training_result, "metrics", {}) or {}
    tracking_uri = metrics.get("mlflow_tracking_uri") or fallback_tracking_uri
    experiment_id = metrics.get("mlflow_experiment_id")
    run_id = metrics.get("mlflow_run_id")
    if isinstance(tracking_uri, str) and isinstance(experiment_id, str) and isinstance(run_id, str):
        return MlflowCoordinates(tracking_uri, experiment_id, run_id)

    if not isinstance(tracking_uri, str):
        return None

    direct_run_id = getattr(training_result, "run_id", None)
    if isinstance(direct_run_id, str) and direct_run_id:
        try:
            resolved_experiment_id = (
                MlflowClient(tracking_uri=tracking_uri).get_run(direct_run_id).info.experiment_id
            )
        except Exception:  # noqa: BLE001
            resolved_experiment_id = None
        if isinstance(resolved_experiment_id, str) and resolved_experiment_id:
            return MlflowCoordinates(tracking_uri, resolved_experiment_id, direct_run_id)

    found = find_mlflow_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    if found is None:
        return None

    fallback_experiment_id, fallback_run_id = found
    return MlflowCoordinates(tracking_uri, fallback_experiment_id, fallback_run_id)


def create_fallback_training_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    tags: Mapping[str, str],
    checkpoint_path: Path,
) -> tuple[str, str]:
    """Create a training run directly through MlflowClient when DLKit leaves none behind."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name)
    )
    run = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.runName": run_name, **dict(tags)},
    )
    upload_checkpoint_artifacts(client, run.info.run_id, checkpoint_path)
    return experiment_id, run.info.run_id


def _log_training_context(
    *,
    tracking_uri: str,
    run_id: str,
    assignment_id: str | None,
    assignment_display_name: str | None,
    resolved_dataset_id: str,
    dataset_display_name: str,
    dataset_registry_id: str | None,
    job_registry_id: str | None,
    job_display_name: str,
) -> None:
    """Log stable ids and display names to the training MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Target MLflow run ID.
        assignment_id: Registry assignment ID, or None.
        assignment_display_name: Human-readable assignment name, or None.
        resolved_dataset_id: Dataset identity resolved from the dataset config's own
            content — not necessarily equal to dataset_registry_id (the [[datasets]]
            lookup key). Logged under the stable "dataset_id" MLflow param key.
        dataset_display_name: Human-readable dataset name.
        dataset_registry_id: Registry dataset ID, or None.
        job_registry_id: Registry job ID, or None.
        job_display_name: Human-readable job name.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    params: dict[str, str] = {
        "dataset_id": resolved_dataset_id,
        "dataset_display_name": dataset_display_name,
        "job_display_name": job_display_name,
    }
    if assignment_id is not None:
        params["assignment_id"] = assignment_id
    if assignment_display_name is not None:
        params["assignment_display_name"] = assignment_display_name
    if dataset_registry_id is not None:
        params["dataset_registry_id"] = dataset_registry_id
    if job_registry_id is not None:
        params["job_registry_id"] = job_registry_id
    for key, value in params.items():
        client.log_param(run_id, key, value)


def _extract_evaluation_arrays(
    all_numpy: Any,
    contract: RuntimeDatasetContract,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract prediction and target arrays from a normalized numpy payload.

    Args:
        all_numpy: Normalized numpy payload dict (predictions + targets).
        contract: Runtime dataset contract for key resolution.

    Returns:
        (y_pred, y_true) arrays, or None if the payload is absent or incomplete.
    """
    if not isinstance(all_numpy, Mapping):
        return None

    predictions_raw = all_numpy.get("predictions")
    targets_raw = all_numpy.get("targets")
    if not isinstance(predictions_raw, Mapping) or not isinstance(targets_raw, Mapping):
        return None

    y_pred_raw = predictions_raw.get(contract.prediction_name)
    y_true_raw = targets_raw.get(contract.target_name)
    if y_pred_raw is None or y_true_raw is None:
        return None

    y_pred = np.asarray(y_pred_raw)
    y_true = np.asarray(y_true_raw)
    if y_pred.size == 0 or y_true.size == 0:
        return None
    return y_pred, y_true


def _normalize_training_numpy_payload(
    all_numpy: Any,
    contract: RuntimeDatasetContract,
) -> Mapping[str, Any] | None:
    """Normalize DLKit prediction payloads into the local runtime contract once.

    Args:
        all_numpy: Raw DLKit numpy payload (may expose 'output' instead of the
            canonical prediction key).
        contract: Runtime dataset contract for key resolution.

    Returns:
        Normalized payload with predictions keyed by contract.prediction_name,
        or None if the payload structure is invalid.

    Raises:
        ValueError: If neither the canonical prediction key nor 'output' is found.
    """
    if not isinstance(all_numpy, Mapping):
        return None

    predictions_raw = all_numpy.get("predictions")
    targets_raw = all_numpy.get("targets")
    if not isinstance(predictions_raw, Mapping) or not isinstance(targets_raw, Mapping):
        return None

    if contract.prediction_name in predictions_raw:
        prediction_value = predictions_raw[contract.prediction_name]
    elif "output" in predictions_raw:
        prediction_value = predictions_raw["output"]
    else:
        raise ValueError(
            "Training prediction payload must expose either the canonical prediction key "
            f"'{contract.prediction_name}' or the DLKit boundary key 'output'."
        )

    normalized = dict(all_numpy)
    normalized["predictions"] = {contract.prediction_name: prediction_value}
    normalized["targets"] = dict(targets_raw)
    return normalized


def _get_normalized_training_numpy_payload(
    training_result: Any,
    contract: RuntimeDatasetContract,
) -> Mapping[str, Any] | None:
    """Read and normalize DLKit numpy payloads when the result exposes them.

    Args:
        training_result: DLKit training result object.
        contract: Runtime dataset contract for key resolution.

    Returns:
        Normalized payload, or None if the result has no to_numpy() method.
    """
    to_numpy = getattr(training_result, "to_numpy", None)
    if not callable(to_numpy):
        return None
    return _normalize_training_numpy_payload(to_numpy(), contract)


def _log_training_evaluation(
    tracking_uri: str,
    run_id: str,
    numpy_payload: Mapping[str, Any] | None,
    figures_dir: Path,
    contract: RuntimeDatasetContract,
) -> None:
    """Compute diagnostics from training predictions and log to an existing MLflow run.

    Uses the predictions and targets already captured by trainer.predict() during
    training. Delegates figure writing and MLflow logging to dedicated helpers.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        numpy_payload: Normalized DLKit prediction/target payload.
        figures_dir: Directory to write the diagnostics figure.
        contract: Runtime dataset contract for array key resolution.
    """
    selected = _extract_evaluation_arrays(numpy_payload, contract)
    if selected is None:
        logger.warning(
            "Skipping training diagnostics logging: unable to resolve prediction/target arrays."
        )
        return

    y_pred, y_true = selected
    try:
        diagnostics = compute_diagnostics(y_pred, y_true)
        figure_path = write_diagnostics_figure(y_true, y_pred, figures_dir)
        log_diagnostics_to_mlflow(tracking_uri, run_id, diagnostics, figure_path)
        metrics_dir = figures_dir.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "training_diagnostics.json").write_text(
            json.dumps({k: float(v) for k, v in diagnostics.metrics.items()}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Skipping MLflow diagnostics logging for run {}: {}",
            run_id,
            exc,
        )


def _stage_training_artifacts(
    *,
    workspace: Any,
    training_result: Any,
    numpy_payload: Mapping[str, Any] | None,
    job_config_path: Path,
    data_config_path: Path | None,
) -> None:
    """Stage full training artifacts into the workspace for MLflow upload.

    Args:
        workspace: Assignment workspace (provides root_dir, predictions_dir).
        training_result: DLKit training result object.
        numpy_payload: Normalized prediction/target payload.
        job_config_path: Path to the job TOML config.
        data_config_path: Path to the dataset TOML config, or None.
    """
    config_dir = workspace.root_dir / "config"
    metrics_dir = workspace.root_dir / "metrics"
    config_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(job_config_path, config_dir / job_config_path.name)
    if data_config_path is not None:
        shutil.copy2(data_config_path, config_dir / data_config_path.name)

    metrics_payload = getattr(training_result, "metrics", {}) or {}
    (metrics_dir / "training_result_metrics.json").write_text(
        json.dumps(coerce_jsonable(metrics_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_training_predictions(
        training_result,
        workspace.predictions_dir,
        numpy_payload=numpy_payload,
    )


def _resolve_mlflow_training_checkpoint(
    *,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
) -> Path:
    """Resolve checkpoint artifacts for a completed MLflow run.

    Args:
        run_id: MLflow run ID whose checkpoints to resolve.
        artifact_leases: Lease manager that owns any temporary materialization.

    Returns:
        Path to the single checkpoint file found under the resolved artifacts.

    Raises:
        RuntimeError: If MLflow artifact resolution fails.
    """
    try:
        checkpoint_root = artifact_leases.resolve_dir(run_id, CHECKPOINT_ARTIFACT_DIR).path
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve checkpoints for run '{run_id}' from MLflow: {exc}"
        ) from exc
    return find_single_checkpoint(checkpoint_root)


def _iter_local_checkpoint_candidates(
    training_result: Any,
    workspace: Any,
) -> Iterator[Path | str | None]:
    """Yield local checkpoint candidates in descending order of authority.

    Lazily evaluated: the workspace directory scans only run once every
    result-supplied candidate ahead of them has been rejected.

    Args:
        training_result: DLKit training result object.
        workspace: Assignment workspace (provides checkpoint_dir, root_dir).

    Yields:
        Each candidate location, which may be unset or may not exist on disk.
    """
    yield getattr(training_result, "checkpoint_path", None)

    artifacts = getattr(training_result, "artifacts", {}) or {}
    for key in (BEST_CHECKPOINT_ARTIFACT_KEY, LAST_CHECKPOINT_ARTIFACT_KEY):
        yield artifacts.get(key)

    yield get_latest_checkpoint(workspace.checkpoint_dir)
    yield get_latest_checkpoint(workspace.root_dir / RETAINED_CHECKPOINTS_DIR_NAME)


def _resolve_local_training_checkpoint(
    *,
    training_result: Any,
    workspace: Any,
) -> Path | None:
    """Resolve the produced checkpoint from local artifacts only.

    Args:
        training_result: DLKit training result object.
        workspace: Assignment workspace (provides checkpoint_dir, root_dir).

    Returns:
        The first candidate from `_iter_local_checkpoint_candidates` that
        exists on disk, or None when no local checkpoint exists.
    """
    for candidate in _iter_local_checkpoint_candidates(training_result, workspace):
        if candidate is None:
            continue
        checkpoint = Path(candidate)
        if checkpoint.exists():
            return checkpoint
    return None


def _resolve_training_checkpoint(
    *,
    training_result: Any,
    workspace: Any,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
) -> Path:
    """Resolve the produced checkpoint from local artifacts or a concrete MLflow lease.

    Args:
        training_result: DLKit training result object.
        workspace: Assignment workspace (provides checkpoint_dir, root_dir).
        run_id: MLflow run ID for artifact fallback.
        artifact_leases: Required lease manager for MLflow artifact fallback.

    Returns:
        Path to the resolved checkpoint file.

    Raises:
        RuntimeError: If no checkpoint is found through any mechanism.
    """
    local_checkpoint = _resolve_local_training_checkpoint(
        training_result=training_result,
        workspace=workspace,
    )
    if local_checkpoint is not None:
        return local_checkpoint

    logger.info(
        "Checkpoint missing locally for run {}. Resolving from MLflow artifacts.",
        run_id,
    )
    return _resolve_mlflow_training_checkpoint(
        run_id=run_id,
        artifact_leases=artifact_leases,
    )
