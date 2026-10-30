"""Application-layer DTOs for the inference use case."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.domain.inference import InferenceOutputs, InferencePredictions

__all__ = [
    "DataSource",
    "InferenceConfig",
    "InferenceData",
    "InferenceOutputs",
    "InferencePredictions",
]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration for the inference workflow.

    Bundles all inference parameters into a single immutable object,
    replacing wide function signatures.

    Args:
        config_path: Path to model configuration file.
        data_config_path: Path to data configuration file (optional).
        checkpoint_path: Explicit checkpoint path (optional).
        features_path: Path to features file (optional).
        targets_path: Path to targets file (optional).
        output_root: Custom output root (optional).
        synthetic_benchmark: Whether to use synthetic benchmark data.
        comparison_config_path: Path to comparison configuration (for synthetic).
    """

    config_path: Path
    data_config_path: Path | None = None
    checkpoint_path: Path | None = None
    features_path: Path | None = None
    targets_path: Path | None = None
    output_root: Path | None = None
    synthetic_benchmark: bool = False
    comparison_config_path: Path | None = None


class DataSource(StrEnum):
    """Where inference data came from."""

    STANDARD = "standard"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class InferenceData:
    """Loaded data for inference.

    Args:
        features: Feature arrays keyed by name (e.g., "rhs", "matrix").
        targets: Target arrays keyed by name (optional for synthetic).
        source: Which loading strategy produced the data.
        metadata: Additional metadata about the data.
    """

    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray] | None
    source: DataSource
    metadata: dict[str, Any]
