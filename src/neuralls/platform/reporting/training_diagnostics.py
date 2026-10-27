"""Pure diagnostic functions for evaluating model predictions.

This module provides stateless, pure functions for computing prediction
error metrics. No side effects — caller decides where to log results.

Functions:
    compute_diagnostics: Compute error metrics from prediction arrays.

Domain models:
    PredictionDiagnostics: Immutable container for computed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PredictionDiagnostics:
    """Immutable container for prediction error metrics.

    Attributes:
        mae: Mean absolute error
        mse: Mean squared error
        duration_seconds: Inference duration (0.0 if unknown)
        metrics: MLflow-ready flat dict of all metrics under 'eval/' prefix
    """

    mae: float
    mse: float
    duration_seconds: float
    metrics: dict[str, float]


def compute_diagnostics(
    y_pred: NDArray,
    y_true: NDArray,
    duration: float = 0.0,
) -> PredictionDiagnostics:
    """Compute error metrics from prediction and target arrays.

    Pure function: no I/O, no side effects.

    Args:
        y_pred: Predicted values, shape (N,) or (N, D)
        y_true: True target values, same shape as y_pred
        duration: Elapsed inference time in seconds

    Returns:
        PredictionDiagnostics with mae, mse, and mlflow-ready metrics
    """
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    true = np.asarray(y_true, dtype=np.float64).ravel()

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    true_std = float(np.std(true))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    # Constant targets make the scale and R² undefined; NaN beats a misleading 0 or inf.
    metrics = {
        "eval/mae": mae,
        "eval/mse": mse,
        "eval/rmse": float(np.sqrt(mse)),
        "eval/r2": float(1.0 - np.sum(diff**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "eval/pred_target_std_ratio": float(np.std(pred)) / true_std
        if true_std > 0
        else float("nan"),
        "eval/mean_shift": float(pred.mean() - true.mean()) / true_std
        if true_std > 0
        else float("nan"),
        "eval/duration_seconds": duration,
    }

    return PredictionDiagnostics(
        mae=mae,
        mse=mse,
        duration_seconds=duration,
        metrics=metrics,
    )
