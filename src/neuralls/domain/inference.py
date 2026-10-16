"""Domain-layer DTOs for inference data flow.

These types cross the application/platform boundary and therefore live in
the domain layer, which both ``application`` and ``platform`` can access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferencePredictions:
    """Prediction results from model inference.

    Args:
        predictions: Predicted arrays keyed by name.
        targets: True target arrays for comparison.
        metadata: Additional metadata about predictions.
    """

    predictions: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InferenceOutputs:
    """Saved outputs from the inference workflow.

    Args:
        csv_paths: Paths to saved CSV prediction files.
        metrics: Computed metrics (e.g., errors, norms).
    """

    csv_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
