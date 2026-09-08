"""Inference helpers shared by CLI scripts and orchestration flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.application.inference.models import InferenceConfig
from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment
from neuralls.platform.config.settings import NeurallsSettings, require_settings

PREDICTION_ARTIFACTS: tuple[str, ...] = ("figures", "predictions")


@dataclass(frozen=True)
class InferenceResult:
    """Local result container for the ``run_inference`` public API.

    Attributes:
        predictions: Predicted values array
        y_true: True target values array
        y_pred: Predicted values array (same as predictions, kept for compatibility)
        duration_seconds: Elapsed time for inference in seconds
        plot_path: Path to parity/residuals plot or None
        diagnostic_plot_path: Path to diagnostics plot or None
    """

    predictions: np.ndarray | None
    y_true: np.ndarray | None
    y_pred: np.ndarray | None
    duration_seconds: float
    plot_path: Path | None
    diagnostic_plot_path: Path | None


def _validate_inference_config(config: InferenceConfig) -> None:
    """Validate inference configuration early.

    Args:
        config: Inference configuration to validate

    Raises:
        ValueError: If checkpoint path not provided

    Note:
        After this function returns, checkpoint_path is guaranteed non-None.
    """
    if config.checkpoint_path is None:
        raise ValueError("No checkpoint path specified")


def _load_assignment_settings(
    config: InferenceConfig,
    settings: NeurallsSettings,
    case_config_path: Path | None = None,
) -> tuple[Any, Any, str]:
    """Load assignment configuration in inference mode.

    Args:
        config: Inference configuration

    Returns:
        Tuple of (settings, workspace, dataset_id)

    Raises:
        ValueError: If data_config_path not provided
    """
    # Guard: Validate data_config_path exists
    if config.data_config_path is None:
        raise ValueError("data_config_path is required for inference")

    dataset_registry_id = Path(config.data_config_path).stem
    assignment = load_assignment(
        config.config_path,
        config.data_config_path,
        settings,
        case_config_path=case_config_path,
        output_root=config.output_root,
        mode="inference",
        identity=AssignmentIdentity(dataset_registry_id=dataset_registry_id),
    )
    settings = assignment.settings
    workspace = assignment.workspace
    dataset_id = workspace.dataset_id
    return settings, workspace, dataset_id


def _execute_inference_pipeline(
    config: InferenceConfig,
    runtime_settings: NeurallsSettings,
    settings: Any,
    workspace: Any,
    dataset_id: str,
) -> tuple[Any, dict[str, float], list[Path]]:
    """Execute core inference pipeline.

    Args:
        config: Inference configuration
        settings: Assignment settings
        workspace: Workspace paths
        dataset_id: Dataset identifier

    Returns:
        Tuple of (predictions, metrics_dict, plot_paths)
    """
    from neuralls.application.inference.prediction import run_prediction
    from neuralls.composition.inference.data_loading import load_inference_data
    from neuralls.platform.dlkit.inference_adapter import (
        create_inference_predictor,
        resolve_batch_size,
    )
    from neuralls.platform.reporting.predictions import (
        save_inference_outputs,
        save_synthetic_predictions,
    )

    # Type guard: checkpoint_path validated before this function is called
    assert config.checkpoint_path is not None

    # Load data (strategy pattern: standard vs synthetic)
    data = load_inference_data(
        workspace=workspace,
        settings=runtime_settings,
        features_path=config.features_path,
        targets_path=config.targets_path,
        synthetic_benchmark=config.synthetic_benchmark,
        comparison_config_path=config.comparison_config_path,
    )

    # Run prediction (transforms applied automatically from checkpoint)
    with create_inference_predictor(config.checkpoint_path, settings) as predictor:
        predictions = run_prediction(predictor, data, batch_size=resolve_batch_size(settings))

    # Save synthetic results separately (if applicable)
    if config.synthetic_benchmark and data.metadata.get("source") == "synthetic":
        from neuralls.platform.storage.filesystem import sanitize_identifier

        run_identifier = sanitize_identifier(str(workspace.run_id or config.checkpoint_path.stem))
        dataset_slug = sanitize_identifier(str(dataset_id))
        identifier = f"{dataset_slug}-{run_identifier}"
        save_synthetic_predictions(predictions, workspace.predictions_dir, identifier)

    # Save outputs (CSV + plots)
    outputs = save_inference_outputs(
        predictions=predictions,
        workspace=workspace,
        settings=settings,
        checkpoint_path=config.checkpoint_path,
        config_path=config.config_path,
        dataset_id=dataset_id,
        save_plots=config.save_plots,
        figures_dir=config.figures_dir,
    )

    return predictions, outputs.metrics, outputs.plot_paths


def _build_inference_result(
    predictions: Any,
    metrics: dict[str, float],
    plot_paths: list[Path],
) -> InferenceResult:
    """Build typed inference result from predictions and metrics.

    Args:
        predictions: Prediction results
        metrics: Performance metrics
        plot_paths: Generated plot paths

    Returns:
        InferenceResult with all outputs
    """
    return InferenceResult(
        predictions=predictions.predictions.get("y_pred"),
        y_true=predictions.targets.get("y_true"),
        y_pred=predictions.predictions.get("y_pred"),
        duration_seconds=metrics.get("duration_seconds", 0.0),
        plot_path=plot_paths[0] if len(plot_paths) > 0 else None,
        diagnostic_plot_path=plot_paths[1] if len(plot_paths) > 1 else None,
    )


def run_inference(
    *,
    config_path: str | Path,
    settings: NeurallsSettings | None = None,
    case_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    features_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    save_plots: bool = True,
    figures_dir: str | Path | None = None,
    enable_mlflow: bool = False,
    output_root: str | Path | None = None,
    synthetic_benchmark: bool = False,
    comparison_config_path: str | Path | None = None,
) -> InferenceResult:
    """Run inference for parity plot generation.

    This is a pure orchestration function that composes single-responsibility
    functions. No business logic here - only composition.

    Args:
        config_path: Path to model configuration file
        data_config_path: Path to data configuration file (optional)
        checkpoint_path: Path to model checkpoint (required)
        features_path: Path to features file (for standard inference)
        targets_path: Path to targets file (for standard inference)
        save_plots: Whether to generate diagnostic plots
        figures_dir: Custom figures directory (optional)
        enable_mlflow: Whether to log to MLflow
        output_root: Custom output root directory (optional)
        synthetic_benchmark: Use synthetic benchmark data
        comparison_config_path: Path to comparison config (for synthetic)

    Returns:
        InferenceResult with all prediction outputs and metadata

    Raises:
        ValueError: If no checkpoint path or no data available
        FileNotFoundError: If required files don't exist
    """
    resolved_case_config_path = Path(case_config_path) if case_config_path else None
    settings = require_settings(settings, case_config_path=resolved_case_config_path)
    from neuralls.platform.reporting.predictions import (
        finalize_mlflow_run,
        start_mlflow_run,
    )

    # Build config object to replace 11 parameters
    config = InferenceConfig(
        config_path=Path(config_path),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        data_config_path=Path(data_config_path) if data_config_path else None,
        features_path=Path(features_path) if features_path else None,
        targets_path=Path(targets_path) if targets_path else None,
        save_plots=save_plots,
        figures_dir=Path(figures_dir) if figures_dir else None,
        enable_mlflow=enable_mlflow,
        output_root=Path(output_root) if output_root else None,
        synthetic_benchmark=synthetic_benchmark,
        comparison_config_path=Path(comparison_config_path) if comparison_config_path else None,
    )

    # 1. Validate configuration
    _validate_inference_config(config)

    # 2. Load assignment settings
    assignment_settings, workspace, dataset_id = _load_assignment_settings(
        config,
        settings,
        case_config_path=resolved_case_config_path,
    )

    # 3. Start MLflow run (if enabled)
    mlflow_state = start_mlflow_run(
        assignment_settings,
        workspace,
        dataset_id,
        config.enable_mlflow,
    )

    metrics: dict[str, float] | None = None
    error: Exception | None = None

    try:
        # 4. Execute inference pipeline
        predictions, metrics, plot_paths = _execute_inference_pipeline(
            config, settings, assignment_settings, workspace, dataset_id
        )

        # 5. Build typed result
        return _build_inference_result(predictions, metrics, plot_paths)

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        error = exc
        raise
    finally:
        # 6. Finalize MLflow run
        if mlflow_state is not None and metrics is not None:
            finalize_mlflow_run(mlflow_state, metrics, workspace)
        elif mlflow_state is not None:
            # Cleanup even if metrics weren't set
            from neuralls.platform.tracking.mlflow import finalize_run

            finalize_run(
                mlflow_state,
                metrics={},
                workspace_root=workspace.root_dir,
                allowlist=PREDICTION_ARTIFACTS,
                failed=error is not None,
            )
