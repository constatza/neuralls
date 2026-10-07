"""Application-layer tests for batch inference prediction orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuralls.application.inference.models import InferenceData
from neuralls.application.inference.prediction import process_predictions, run_prediction
from neuralls.domain.inference_ports import InferencePredictorPort


class FakeInferencePredictor(InferencePredictorPort):
    """Predictor port stub that returns pre-seeded numpy batches."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = list(outputs)
        self.cleaned_up = False
        self.seen_batches: list[dict[str, np.ndarray]] = []

    def predict_batch(self, feature_batch: dict[str, np.ndarray]) -> np.ndarray:
        self.seen_batches.append({name: values.copy() for name, values in feature_batch.items()})
        if not self._outputs:
            raise ValueError("No seeded prediction outputs remaining.")
        return self._outputs.pop(0)

    def cleanup(self) -> None:
        self.cleaned_up = True


@dataclass(frozen=True)
class InferenceFixtures:
    data: InferenceData
    expected_batches: list[np.ndarray]
    expected_predictions: np.ndarray


@pytest.fixture
def multi_feature_inference_fixtures() -> InferenceFixtures:
    features = {
        "x": np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64),
        "rhs": np.array([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=np.float64),
    }
    targets = {"y_true": np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)}
    data = InferenceData(
        features=features,
        targets=targets,
        metadata={"source": "fixture"},
    )
    return InferenceFixtures(
        data=data,
        expected_batches=[
            np.array([[1.0], [2.0]], dtype=np.float64),
            np.array([[3.0], [4.0]], dtype=np.float64),
            np.array([[5.0]], dtype=np.float64),
        ],
        expected_predictions=np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64),
    )


@pytest.fixture
def scalar_prediction_batches() -> list[np.ndarray]:
    return [np.array(1.5, dtype=np.float64), np.array(2.5, dtype=np.float64)]


@pytest.fixture
def inconsistent_prediction_batches() -> list[np.ndarray]:
    return [
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([3.0, 4.0], dtype=np.float64),
    ]


@pytest.fixture
def empty_inference_data() -> InferenceData:
    return InferenceData(
        features={"x": np.empty((0, 1), dtype=np.float64)},
        targets={"y_true": np.empty((0,), dtype=np.float64)},
        metadata={"source": "empty"},
    )


def test_run_prediction_preserves_batch_order(
    multi_feature_inference_fixtures: InferenceFixtures,
) -> None:
    predictor = FakeInferencePredictor(
        outputs=[
            np.array([[10.0], [11.0]], dtype=np.float64),
            np.array([[12.0], [13.0]], dtype=np.float64),
            np.array([[14.0]], dtype=np.float64),
        ]
    )

    result = run_prediction(
        predictor,
        multi_feature_inference_fixtures.data,
        batch_size=2,
    )

    assert_array_equal(
        result.predictions["y_pred"],
        multi_feature_inference_fixtures.expected_predictions,
    )
    assert multi_feature_inference_fixtures.data.targets is not None
    assert_array_equal(
        result.targets["y_true"],
        multi_feature_inference_fixtures.data.targets["y_true"],
    )
    assert len(predictor.seen_batches) == 3
    for batch, expected in zip(
        predictor.seen_batches,
        multi_feature_inference_fixtures.expected_batches,
        strict=True,
    ):
        assert_array_equal(batch["x"], expected)


def test_run_prediction_measures_a_real_duration(
    multi_feature_inference_fixtures: InferenceFixtures,
) -> None:
    """`duration_seconds` is a real wall-clock measurement, not the old hardcoded 0.0."""

    class SlowPredictor(FakeInferencePredictor):
        def predict_batch(self, feature_batch: dict[str, np.ndarray]) -> np.ndarray:
            time.sleep(0.01)
            return super().predict_batch(feature_batch)

    predictor = SlowPredictor(
        outputs=[
            np.array([[10.0], [11.0]], dtype=np.float64),
            np.array([[12.0], [13.0]], dtype=np.float64),
            np.array([[14.0]], dtype=np.float64),
        ]
    )

    result = run_prediction(
        predictor,
        multi_feature_inference_fixtures.data,
        batch_size=2,
    )

    assert result.metadata["duration_seconds"] > 0.0


def test_process_predictions_normalizes_scalar_batches(
    scalar_prediction_batches: list[np.ndarray],
) -> None:
    result = process_predictions(scalar_prediction_batches)

    assert_array_equal(result, np.array([1.5, 2.5], dtype=np.float64))


def test_process_predictions_rejects_inconsistent_batch_shapes(
    inconsistent_prediction_batches: list[np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="Inconsistent prediction batch shapes"):
        process_predictions(inconsistent_prediction_batches)


def test_run_prediction_rejects_empty_prediction_stream(
    empty_inference_data: InferenceData,
) -> None:
    predictor = FakeInferencePredictor(outputs=[])

    with pytest.raises(ValueError, match="Predictor returned no predictions"):
        run_prediction(predictor, empty_inference_data, batch_size=2)
