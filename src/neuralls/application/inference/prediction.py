"""Prediction execution logic for inference workflow.

This module provides pure functions for running model predictions,
including batch processing and result collection.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from loguru import logger
from torchalg.utils.device import resolve_device

from neuralls.application.inference.models import InferenceData, InferencePredictions
from neuralls.domain.inference_ports import InferencePredictorPort
from neuralls.shared.device import track_resource_usage


def iterate_feature_batches(
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield contiguous feature batches for streaming prediction.

    Args:
        feature_arrays: Feature arrays keyed by name.
        batch_size: Number of samples per batch.

    Yields:
        Batched feature dictionaries.
    """
    total = next(iter(feature_arrays.values())).shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {name: array[start:end] for name, array in feature_arrays.items()}


def collect_predictions(
    predictor: InferencePredictorPort,
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[list[np.ndarray], float]:
    """Run inference for every feature batch.

    Args:
        predictor: Batch predictor port implementation.
        feature_arrays: Feature arrays keyed by name.
        batch_size: Number of samples per batch.

    Returns:
        Tuple of (predictions_list, duration_seconds) — the real wall-clock
        time spent in `predictor.predict_batch()` across every batch.

    Raises:
        ValueError: If predictor returns no predictions.
    """
    predictions: list[np.ndarray] = []
    with track_resource_usage(resolve_device()) as usage:
        for batch in iterate_feature_batches(feature_arrays, batch_size):
            predictions.append(np.asarray(predictor.predict_batch(batch), dtype=np.float64))
    if not predictions:
        raise ValueError("Predictor returned no predictions.")
    return predictions, usage().wall_time_seconds


def stack_predictions(raw_predictions: list[np.ndarray]) -> np.ndarray:
    """Concatenate raw prediction batches into one array, preserving row shape.

    Unlike `process_predictions`, this does not flatten the result — use it
    when callers need per-row vectors intact (e.g. a POD-2G snapshot ensemble,
    shape `(n_samples, n_dofs)`), rather than a flat scalar-per-row array.

    Args:
        raw_predictions: List of batch prediction arrays from predictor.

    Returns:
        Concatenated prediction array, one row per input sample.
    """
    if not raw_predictions:
        raise ValueError("No predictions available to process.")

    normalized: list[np.ndarray] = []
    expected_tail_shape: tuple[int, ...] | None = None
    for prediction in raw_predictions:
        array = np.asarray(prediction, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.shape[0] == 0:
            raise ValueError("Predictor returned an empty prediction batch.")

        tail_shape = array.shape[1:]
        if expected_tail_shape is None:
            expected_tail_shape = tail_shape
        elif tail_shape != expected_tail_shape:
            raise ValueError(
                "Inconsistent prediction batch shapes: "
                f"expected trailing shape {expected_tail_shape}, got {tail_shape}"
            )

        normalized.append(array)

    return np.concatenate(normalized, axis=0)


def process_predictions(
    raw_predictions: list[np.ndarray],
) -> np.ndarray:
    """Process raw prediction batches into a single flattened array.

    Args:
        raw_predictions: List of batch prediction arrays from predictor.

    Returns:
        Flattened prediction array.
    """
    return stack_predictions(raw_predictions).ravel()


def run_prediction(
    predictor: InferencePredictorPort,
    data: InferenceData,
    batch_size: int = 256,
) -> InferencePredictions:
    """Execute prediction on inference data.

    Transforms are applied automatically via checkpoint-based transforms
    (enabled with apply_transforms=True in predictor creation).

    Args:
        predictor: Batch predictor port implementation.
        data: Inference data with features and targets.
        batch_size: Batch size for inference (defaults to 256).

    Returns:
        InferencePredictions with results and metadata.
    """
    feature_arrays = data.features

    logger.info(f"Running inference on {data.metadata.get('source', 'unknown')} data...")
    raw_preds, duration = collect_predictions(predictor, feature_arrays, batch_size)

    predictions = process_predictions(raw_preds)

    targets = next(iter(data.targets.values())) if data.targets else np.zeros_like(predictions)

    return InferencePredictions(
        predictions={"y_pred": predictions},
        targets={"y_true": targets},
        metadata={
            **data.metadata,
            "duration_seconds": duration,
            "num_predictions": len(predictions),
        },
    )
