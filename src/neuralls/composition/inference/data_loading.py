"""Data loading strategies for inference workflow.

This module provides pure functions for loading inference data from various sources:
- Standard data from feature/target files
- Synthetic benchmark data from comparison configurations
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dlkit.io import load_array
from loguru import logger

from neuralls.application.inference.models import DataSource, InferenceData
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.reporting.synthetic import generate_synthetic_test_case
from neuralls.shared.constants import RHS_ZARR_FILENAME, SOLUTIONS_ZARR_FILENAME


def validate_data_availability(
    standard_available: bool,
    synthetic_available: bool,
) -> None:
    """Validate that at least one data source is available.

    Args:
        standard_available: Whether standard data is available
        synthetic_available: Whether synthetic data is available

    Raises:
        ValueError: If no data source is available
    """
    if not standard_available and not synthetic_available:
        raise ValueError(
            "No data available for inference. "
            "Provide either standard data files or enable synthetic benchmark."
        )


def resolve_standard_data_paths(
    workspace: AssignmentWorkspace,
    features_path: Path | None,
    targets_path: Path | None,
) -> tuple[Path, Path]:
    """Resolve paths for standard feature and target files.

    Args:
        workspace: Assignment workspace with default data directory
        features_path: Explicit features path (optional)
        targets_path: Explicit targets path (optional)

    Returns:
        Tuple of (features_path, targets_path)
    """
    feat_path = Path(features_path) if features_path else workspace.data_dir / RHS_ZARR_FILENAME
    tgt_path = Path(targets_path) if targets_path else workspace.data_dir / SOLUTIONS_ZARR_FILENAME
    return feat_path, tgt_path


def load_standard_data(
    workspace: AssignmentWorkspace,
    features_path: Path | None,
    targets_path: Path | None,
) -> InferenceData | None:
    """Load standard inference data from feature and target files.

    Args:
        workspace: Assignment workspace with default data directory
        features_path: Path to features file (optional)
        targets_path: Path to targets file (optional)

    Returns:
        InferenceData object or None if files don't exist

    Note:
        - Defaults to workspace.data_dir/rhs.npy and workspace.data_dir/solutions.npy
        - Uses explicit array files under the dataset artifact layout
        - Returns None if files don't exist (allows fallback to synthetic)
    """
    contract = default_training_dataset_contract()
    feat_path, tgt_path = resolve_standard_data_paths(workspace, features_path, targets_path)

    if not feat_path.exists() or not tgt_path.exists():
        logger.warning(
            f"Standard data files not found at {feat_path} or {tgt_path}. "
            "Skipping standard inference."
        )
        return None

    logger.debug(f"Loading standard features from: {feat_path}")
    features = np.asarray(load_array(feat_path))

    logger.debug(f"Loading standard targets from: {tgt_path}")
    targets = np.asarray(load_array(tgt_path))

    # Ensure targets are 1D if shape is (N, 1)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.ravel()

    # Store as dict for compatibility with DLKit feature entry system
    feature_dict = {contract.primary_input_name: features}
    target_dict = {contract.target_name: targets}

    return InferenceData(
        features=feature_dict,
        targets=target_dict,
        source=DataSource.STANDARD,
        metadata={
            "features_path": str(feat_path),
            "targets_path": str(tgt_path),
            "num_samples": features.shape[0],
        },
    )


def load_synthetic_data(
    comparison_config_path: Path,
    workspace: AssignmentWorkspace,
    settings: NeurallsSettings,
) -> InferenceData:
    """Generate synthetic benchmark data from comparison configuration.

    Synthetic data uses a known solution (x_true = ones) to generate RHS: b = A @ x_true.
    This provides ground truth for evaluating prediction accuracy.

    Args:
        comparison_config_path: Path to comparison configuration file
        workspace: Assignment workspace (for metadata)

    Returns:
        InferenceData object with synthetic features and targets

    Raises:
        ValueError: If comparison config doesn't specify matrix_path
        FileNotFoundError: If matrix file doesn't exist
    """
    contract = default_training_dataset_contract()
    logger.info("Preparing SYNTHETIC benchmark data (x_true = ones).")

    comparison_cfg = load_comparison_config(comparison_config_path, settings)
    matrix_path = Path(comparison_cfg.general.data.matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    # Generate: A, x_true, b = A @ x_true
    _, targets_raw, features_raw = generate_synthetic_test_case(matrix_path)

    # Reshape to (1, N) for single RHS vector
    # Inference expects (num_samples, num_features)
    features = features_raw.reshape(1, -1)
    targets = targets_raw.reshape(1, -1)

    # Store as dict for compatibility with DLKit feature entry system
    feature_dict = {contract.primary_input_name: features}
    target_dict = {contract.target_name: targets}

    return InferenceData(
        features=feature_dict,
        targets=target_dict,
        source=DataSource.SYNTHETIC,
        metadata={
            "matrix_path": str(matrix_path),
            "comparison_config_path": str(comparison_config_path),
            "num_samples": 1,
            "system_size": features.shape[1],
        },
    )


def load_inference_data(
    workspace: AssignmentWorkspace,
    settings: NeurallsSettings,
    features_path: Path | None = None,
    targets_path: Path | None = None,
    synthetic_benchmark: bool = False,
    comparison_config_path: Path | None = None,
) -> InferenceData:
    """Load inference data using appropriate strategy.

    Supports two loading strategies:
    1. Standard: Load from feature/target files
    2. Synthetic: Generate from comparison configuration

    Priority:
    - If synthetic_benchmark=True, loads synthetic data
    - Otherwise, loads standard data
    - Falls back to other source if primary fails

    Args:
        workspace: Assignment workspace
        features_path: Path to features file (for standard)
        targets_path: Path to targets file (for standard)
        synthetic_benchmark: Whether to use synthetic data
        comparison_config_path: Path to comparison config (for synthetic)

    Returns:
        InferenceData object from appropriate source

    Raises:
        ValueError: If no data source is available
    """
    standard_data = None
    synthetic_data = None

    # Load standard data if paths exist
    if not synthetic_benchmark:
        standard_data = load_standard_data(workspace, features_path, targets_path)

    # Load synthetic data if requested
    if synthetic_benchmark:
        if comparison_config_path is None:
            raise ValueError("Comparison config path is required for synthetic benchmark.")
        synthetic_data = load_synthetic_data(comparison_config_path, workspace, settings)

    # Validate availability
    validate_data_availability(
        standard_available=standard_data is not None,
        synthetic_available=synthetic_data is not None,
    )

    # Return primary data source
    if synthetic_benchmark and synthetic_data is not None:
        return synthetic_data
    if standard_data is not None:
        return standard_data
    if synthetic_data is not None:
        return synthetic_data

    # Should never reach here due to validation
    raise ValueError("No data available for inference")
