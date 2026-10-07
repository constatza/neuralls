"""Boundary tests for DLKit-backed prediction adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dlkit.common.errors import DLKitError
from numpy.testing import assert_array_equal

from neuralls.platform.dlkit.inference_adapter import (
    DLKitInferencePredictor,
    create_inference_predictor,
)
from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter, DLKitPredictor


@dataclass(frozen=True)
class FakePredictionOutput:
    predictions: torch.Tensor


class FakeCheckpointPredictor:
    """Minimal predictor stub for adapter boundary tests."""

    def __init__(self, result: Any, feature_names: tuple[str, ...] = ()) -> None:
        self._result = result
        self.feature_names = feature_names
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.closed = False
        self.close_count = 0

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._result

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.closed = True
        self.close_count += 1


class FakeUnloadCheckpointPredictor(FakeCheckpointPredictor):
    """Predictor stub exposing DLKit's explicit unload lifecycle."""

    def unload(self) -> None:
        self.closed = True
        self.close_count += 1


class FakeModel(torch.nn.Module):
    """Minimal model stub exposing one parameter for device discovery."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))


class FakeLoadedPredictor(FakeCheckpointPredictor):
    """Loaded predictor stub with the attributes the factories rely on."""

    def __init__(self, result: Any, model: torch.nn.Module | None = None) -> None:
        super().__init__(result)
        self.model = model if model is not None else FakeModel()


@pytest.fixture
def residual_vector() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)


@pytest.fixture
def feature_batch() -> dict[str, np.ndarray]:
    return {
        "x": np.array([[1.0], [2.0]], dtype=np.float64),
        "rhs": np.array([[3.0], [4.0]], dtype=np.float64),
    }


@pytest.fixture
def prediction_tensor() -> torch.Tensor:
    return torch.tensor([[7.0], [8.0]], dtype=torch.float64)


@pytest.fixture
def solver_prediction_tensor() -> torch.Tensor:
    return torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64)


def _make_supported_result(shape_tensor: torch.Tensor, variant: str) -> Any:
    match variant:
        case "prediction_output":
            return FakePredictionOutput(predictions=shape_tensor)
        case _:
            raise ValueError(f"Unsupported test variant: {variant}")


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_dlkit_predictor_accepts_supported_result_shapes(
    variant: str,
    residual_vector: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(solver_prediction_tensor, variant)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    result = predictor.apply(residual_vector)

    assert_array_equal(result, np.array([4.0, 5.0, 6.0], dtype=np.float64))
    assert len(checkpoint_predictor.calls) == 1
    args, kwargs = checkpoint_predictor.calls[0]
    assert len(args) == 1
    assert kwargs == {}
    assert args[0].dtype == torch.float64


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_dlkit_inference_predictor_accepts_supported_result_shapes(
    variant: str,
    feature_batch: dict[str, np.ndarray],
    prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(prediction_tensor, variant)
    )
    predictor = DLKitInferencePredictor(checkpoint_predictor, device="cpu")

    result = predictor.predict_batch(feature_batch)

    assert_array_equal(result, np.array([[7.0], [8.0]], dtype=np.float64))
    assert len(checkpoint_predictor.calls) == 1
    args, kwargs = checkpoint_predictor.calls[0]
    assert args == ()
    assert set(kwargs) == {"x", "rhs"}
    assert all(value.dtype == torch.float64 for value in kwargs.values())


def test_dlkit_predictor_cleanup_unloads_checkpoint_once(
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeUnloadCheckpointPredictor(
        FakePredictionOutput(predictions=solver_prediction_tensor)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    # Exercise idempotency: owner cleanup and context/GC cleanup can both run.
    predictor.cleanup()
    predictor.cleanup()
    predictor.__exit__(None, None, None)

    assert checkpoint_predictor.closed is True
    assert checkpoint_predictor.close_count == 1


def test_dlkit_predictor_cleanup_falls_back_to_context_exit(
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        FakePredictionOutput(predictions=solver_prediction_tensor)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    # Exercise idempotency for structural predictors without explicit unload().
    predictor.cleanup()
    predictor.cleanup()

    assert checkpoint_predictor.closed is True
    assert checkpoint_predictor.close_count == 1


def test_dlkit_inference_predictor_cleanup_unloads_checkpoint_once(
    prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeUnloadCheckpointPredictor(
        FakePredictionOutput(predictions=prediction_tensor)
    )
    predictor = DLKitInferencePredictor(checkpoint_predictor, device="cpu")

    # Exercise idempotency: owner cleanup and context/GC cleanup can both run.
    predictor.cleanup()
    predictor.cleanup()
    predictor.__exit__(None, None, None)

    assert checkpoint_predictor.closed is True
    assert checkpoint_predictor.close_count == 1


def test_dlkit_predictor_reports_unsupported_result_type(
    residual_vector: torch.Tensor,
) -> None:
    predictor = DLKitPredictor(FakeCheckpointPredictor(object()), device="cpu")

    with pytest.raises(
        RuntimeError,
        match="Unsupported predictor output type: object. Expected an object with a 'predictions' tensor.",
    ):
        predictor.apply(residual_vector)


def test_dlkit_inference_predictor_reports_unsupported_result_type(
    feature_batch: dict[str, np.ndarray],
) -> None:
    predictor = DLKitInferencePredictor(FakeCheckpointPredictor(object()), device="cpu")

    with pytest.raises(
        RuntimeError,
        match="Unsupported predictor output type: object. Expected an object with a 'predictions' tensor.",
    ):
        predictor.predict_batch(feature_batch)


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("checkpoint")
    return checkpoint


def test_solver_factory_uses_dlkit_structural_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    solver_prediction_tensor: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    loaded_predictor = FakeLoadedPredictor(
        FakePredictionOutput(predictions=solver_prediction_tensor)
    )
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args: Any, **kwargs: Any) -> FakeLoadedPredictor:
        del args
        captured_kwargs.update(kwargs)
        return loaded_predictor

    monkeypatch.setattr(
        "neuralls.platform.dlkit.predictor_adapter.load_model",
        fake_load_model,
    )

    predictor = DLKitAdapter().create_predictor(checkpoint_path)

    result = predictor.apply(residual_vector)

    assert_array_equal(result, np.array([4.0, 5.0, 6.0], dtype=np.float64))
    assert len(loaded_predictor.calls) == 1
    assert captured_kwargs["apply_transforms"] is True


def test_inference_factory_uses_dlkit_structural_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    prediction_tensor: torch.Tensor,
    feature_batch: dict[str, np.ndarray],
) -> None:
    loaded_predictor = FakeLoadedPredictor(FakePredictionOutput(predictions=prediction_tensor))

    monkeypatch.setattr(
        "neuralls.platform.dlkit.inference_adapter.load_model",
        lambda *args, **kwargs: loaded_predictor,
    )

    predictor = create_inference_predictor(checkpoint_path)
    result = predictor.predict_batch(feature_batch)

    assert_array_equal(result, np.array([[7.0], [8.0]], dtype=np.float64))
    assert len(loaded_predictor.calls) == 1


@pytest.fixture
def matrix_array() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64)


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_apply_no_extras_uses_positional_call(
    variant: str,
    residual_vector: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(solver_prediction_tensor, variant)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    predictor.apply(residual_vector)

    args, kwargs = checkpoint_predictor.calls[0]
    assert len(args) == 1
    assert len(kwargs) == 0


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_apply_with_extras_uses_keyword_call(
    variant: str,
    residual_vector: torch.Tensor,
    matrix_array: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(solver_prediction_tensor, variant)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    predictor.apply(residual_vector, matrix=matrix_array)

    args, kwargs = checkpoint_predictor.calls[0]
    assert len(args) == 0
    assert "x" in kwargs
    assert "matrix" in kwargs


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_extra_tensor_has_batch_dimension(
    variant: str,
    residual_vector: torch.Tensor,
    matrix_array: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(solver_prediction_tensor, variant)
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    predictor.apply(residual_vector, matrix=matrix_array)

    _, kwargs = checkpoint_predictor.calls[0]
    assert kwargs["matrix"].shape == (1, 3, 3)
    assert kwargs["x"].shape == (1, 3)


@pytest.mark.parametrize("variant", ["prediction_output"])
def test_apply_with_deeponet_feature_names_uses_checkpoint_primary_name(
    variant: str,
    residual_vector: torch.Tensor,
    matrix_array: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        _make_supported_result(solver_prediction_tensor, variant),
        feature_names=("branch", "trunk"),
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    predictor.apply(residual_vector, trunk=matrix_array)

    args, kwargs = checkpoint_predictor.calls[0]
    assert len(args) == 0
    assert "branch" in kwargs
    assert "trunk" in kwargs
    assert "x" not in kwargs


def test_apply_with_mismatched_feature_names_raises_clear_runtime_error(
    residual_vector: torch.Tensor,
    matrix_array: torch.Tensor,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    checkpoint_predictor = FakeCheckpointPredictor(
        FakePredictionOutput(predictions=solver_prediction_tensor),
        feature_names=("branch", "trunk"),
    )
    predictor = DLKitPredictor(checkpoint_predictor, device="cpu")

    with pytest.raises(RuntimeError, match="Cannot resolve primary input name"):
        predictor.apply(residual_vector, matrix=matrix_array)


class FakeForwardContractError(DLKitError):
    """Stand-in for dlkit.common.errors.ForwardContractError in adapter tests."""


class RaisingCheckpointPredictor(FakeCheckpointPredictor):
    """Predictor stub whose predict() raises a dlkit-native contract error."""

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise FakeForwardContractError("predict() received unexpected keyword argument(s)")


def test_apply_translates_dlkit_forward_contract_error_to_runtime_error(
    residual_vector: torch.Tensor,
) -> None:
    predictor = DLKitPredictor(RaisingCheckpointPredictor(None), device="cpu")

    with pytest.raises(RuntimeError, match="Unexpected error during neural inference"):
        predictor.apply(residual_vector)


def test_predict_batch_translates_dlkit_forward_contract_error_to_runtime_error(
    feature_batch: dict[str, np.ndarray],
) -> None:
    predictor = DLKitInferencePredictor(RaisingCheckpointPredictor(None), device="cpu")

    with pytest.raises(RuntimeError, match="Unexpected error during inference batch prediction"):
        predictor.predict_batch(feature_batch)


def test_solver_factory_fails_when_loaded_predictor_has_no_model(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    solver_prediction_tensor: torch.Tensor,
) -> None:
    loaded_predictor = FakeLoadedPredictor(
        FakePredictionOutput(predictions=solver_prediction_tensor),
        model=None,
    )
    loaded_predictor.model = None

    monkeypatch.setattr(
        "neuralls.platform.dlkit.predictor_adapter.load_model",
        lambda *args, **kwargs: loaded_predictor,
    )

    with pytest.raises(RuntimeError, match="Failed to load model from checkpoint"):
        DLKitAdapter().create_predictor(checkpoint_path)


def test_inference_factory_fails_when_predictions_field_is_not_tensor(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    feature_batch: dict[str, np.ndarray],
) -> None:
    @dataclass(frozen=True)
    class BadPredictionOutput:
        predictions: object

    loaded_predictor = FakeLoadedPredictor(BadPredictionOutput(predictions=[1.0, 2.0]))

    monkeypatch.setattr(
        "neuralls.platform.dlkit.inference_adapter.load_model",
        lambda *args, **kwargs: loaded_predictor,
    )

    predictor = create_inference_predictor(checkpoint_path)
    with pytest.raises(
        RuntimeError,
        match="Expected 'predictions' to be a torch.Tensor",
    ):
        predictor.predict_batch(feature_batch)
