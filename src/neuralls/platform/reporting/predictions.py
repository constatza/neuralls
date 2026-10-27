"""Output saving for the inference workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from neuralls.domain.inference import InferenceOutputs, InferencePredictions
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.reporting.prediction_csv import save_prediction_samples_to_csv
from neuralls.platform.reporting.synthetic import save_synthetic_results
from neuralls.platform.storage.filesystem import derive_model_identifier, sanitize_identifier


def derive_run_identifier(
    settings: Any,
    workspace: AssignmentWorkspace,
    checkpoint_path: Path,
    config_path: Path,
) -> str:
    """Derive identifier for this inference run.

    Priority:
    1. Checkpoint parent directory (if named appropriately)
    2. Checkpoint stem
    3. Workspace run_id (if not default)
    4. Model identifier from settings

    Args:
        settings: DLKit workflow settings object
        workspace: Assignment workspace
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file

    Returns:
        Sanitized run identifier string
    """
    # Try checkpoint path
    if checkpoint_path:
        parent = checkpoint_path.parent
        if parent.name.lower() == "checkpoints" and parent.parent.name:
            return parent.parent.name
        if checkpoint_path.stem:
            return checkpoint_path.stem

    # Try workspace run_id
    run_id = workspace.run_id
    if isinstance(run_id, str) and run_id and not run_id.lower().startswith("dlkit-session"):
        return run_id

    # Fallback to model identifier
    return derive_model_identifier(settings, workspace, config_path)


def save_predictions_csv(
    predictions: InferencePredictions,
    output_dir: Path,
    identifier: str,
) -> list[Path]:
    """Save predictions to CSV files.

    Args:
        predictions: Inference predictions with targets
        output_dir: Directory for saving CSVs
        identifier: Filename prefix/identifier

    Returns:
        List of saved CSV paths
    """
    y_true = predictions.targets.get("y_true", np.array([]))
    y_pred = predictions.predictions.get("y_pred", np.array([]))

    csv_paths = save_prediction_samples_to_csv(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        filename_prefix=identifier,
    )

    if csv_paths:
        logger.info(f"Saved prediction samples to CSV: {[str(p) for p in csv_paths]}")

    return csv_paths or []


def save_synthetic_predictions(
    predictions: InferencePredictions,
    output_dir: Path,
    identifier: str,
) -> None:
    """Save synthetic benchmark predictions.

    Args:
        predictions: Inference predictions with synthetic targets
        output_dir: Directory for saving results
        identifier: Filename prefix/identifier
    """
    y_true = predictions.targets.get("y_true", np.array([]))
    y_pred = predictions.predictions.get("y_pred", np.array([]))

    save_synthetic_results(
        output_dir,
        y_true=y_true,
        y_pred=y_pred,
        identifier=identifier,
    )


def save_inference_outputs(
    predictions: InferencePredictions,
    workspace: AssignmentWorkspace,
    settings: Any,
    checkpoint_path: Path,
    config_path: Path,
    dataset_id: str,
) -> InferenceOutputs:
    """Save all inference outputs to workspace.

    Args:
        predictions: Inference predictions with targets
        workspace: Assignment workspace
        settings: DLKit workflow settings object
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        dataset_id: Dataset identifier

    Returns:
        InferenceOutputs with saved paths and metrics
    """
    # Derive identifier
    run_identifier = sanitize_identifier(
        derive_run_identifier(settings, workspace, checkpoint_path, config_path)
    )
    dataset_slug = sanitize_identifier(str(dataset_id))
    identifier = f"{dataset_slug}-{run_identifier}"

    # Save CSV predictions
    csv_paths = save_predictions_csv(predictions, workspace.predictions_dir, identifier)

    # Extract metrics
    duration = predictions.metadata.get("duration_seconds", 0.0)
    metrics = {"duration_seconds": float(duration)}

    return InferenceOutputs(
        csv_paths=csv_paths,
        metrics=metrics,
    )
