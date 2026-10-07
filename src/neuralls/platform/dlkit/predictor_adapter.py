"""Framework-specific adapters for neural predictors.

This module implements adapters that convert framework-specific APIs
(DLKit, PyTorch) to the framework-agnostic PredictorPort interface.

Design Principles:
    - Adapter pattern: Framework-specific code isolated here
    - Resource management: Context managers ensure GPU cleanup
    - Error boundaries: Catch framework errors, raise domain errors
    - Separation: Actions (I/O, GPU) separated from domain logic
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from dlkit import load_model
from dlkit.common.errors import DLKitError
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from loguru import logger
from torchalg.preconditioners.ports import (
    ExtraInputPredictorPort,
    PredictorAdapter,
)

from ._prediction_outputs import extract_prediction_tensor

if TYPE_CHECKING:
    from dlkit.interfaces.inference import CheckpointPredictor as DLKitCheckpointPredictor


def _extra_to_tensors(
    extra_inputs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Move extra input tensors to the predictor device and add a batch axis."""
    return {name: _prepare_model_input(arr, device) for name, arr in extra_inputs.items()}


def _prepare_model_input(value: torch.Tensor, device: str) -> torch.Tensor:
    """Prepare a solver tensor for DLKit prediction."""
    return value.detach().to(device=device, dtype=torch.float64).unsqueeze(0)


def _extract_model_output(value: torch.Tensor) -> torch.Tensor:
    """Remove DLKit's batch dimension while preserving a tensor result."""
    result = value.detach().to(dtype=torch.float64)
    return result.squeeze(0) if result.ndim > 1 and result.shape[0] == 1 else result


def _resolve_primary_input_name(
    feature_names: tuple[str, ...],
    extra_input_keys: frozenset[str],
) -> str:
    """Resolve the forward()-kwarg name for the primary (residual) tensor.

    The primary input is whichever checkpoint-declared feature name is not
    already bound by the caller's extra inputs for this call. Falls back to
    "x" only for legacy checkpoints with no persisted feature_names metadata.

    Args:
        feature_names: Ordered input names restored from checkpoint metadata
            (CheckpointPredictor.feature_names). Empty for legacy checkpoints.
        extra_input_keys: Names of the extra inputs already bound this call
            (e.g. {"trunk"} for a DeepONet call).

    Returns:
        The kwarg name to bind the primary/residual tensor under.

    Raises:
        ValueError: feature_names is non-empty but excluding
            extra_input_keys leaves zero or more than one candidate name —
            a mismatch between the checkpoint's declared inputs and the
            extra inputs bound (e.g. misconfigured extra_input_names).
    """
    if not feature_names:
        return "x"

    candidates = [name for name in feature_names if name not in extra_input_keys]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Cannot resolve primary input name: checkpoint feature_names={feature_names!r} "
        f"vs bound extra inputs {sorted(extra_input_keys)!r} left {len(candidates)} "
        "candidate(s) (expected exactly 1). Check extra_input_names against the "
        "checkpoint's actual forward() parameters."
    )


class DLKitPredictor(ExtraInputPredictorPort):
    """DLKit-based predictor with lifecycle management.

    Implements ExtraInputPredictorPort using DLKit framework. Wraps
    CheckpointPredictor and delegates lifecycle management to it via context
    manager.

    Lifecycle:
        `cleanup()` and context-manager exit both release the wrapped DLKit
        CheckpointPredictor exactly once. This satisfies torchalg's predictor
        port without leaking DLKit lifecycle details into solver code.

    Error Handling:
        - GPU OOM → RuntimeError with helpful message
        - Device mismatch → RuntimeError
        - Model failure → RuntimeError with context
    """

    def __init__(
        self,
        predictor: DLKitCheckpointPredictor,
        device: str,
        required_inputs: tuple[str, ...] = (),
    ) -> None:
        """Initialize predictor with loaded CheckpointPredictor.

        Args:
            predictor: DLKit CheckpointPredictor (already loaded, eval mode)
            device: Device string ("cpu", "cuda", "mps")
            required_inputs: Names of extra arrays the model expects beyond the residual.
                Derived from the model config by the adapter; eliminates TOML duplication.
        """
        self._predictor: DLKitCheckpointPredictor = predictor
        self._device: str = device
        self._required_inputs: tuple[str, ...] = required_inputs
        self._closed = False

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Names of extra arrays this model expects beyond the residual."""
        # ponytail: parse from DLKit config_path when the config schema is known;
        # currently populated by DLKitAdapter from config if available, else empty.
        return self._required_inputs

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Apply neural network to residual.

        Args:
            residual: Numpy residual vector.
            **extra_inputs: Optional named extra arrays forwarded as named tensors.
                When absent, falls back to positional predict(tensor) for backward
                compatibility with models trained without named inputs.

        Returns:
            Predicted correction (float64 numpy)

        Raises:
            RuntimeError: If predictor unloaded or GPU error
        """
        try:
            input_tensor = _prepare_model_input(residual, self._device)
            if extra_inputs:
                primary_name = _resolve_primary_input_name(
                    self._predictor.feature_names, frozenset(extra_inputs)
                )
                tensors: dict[str, torch.Tensor] = {primary_name: input_tensor}
                tensors.update(_extra_to_tensors(extra_inputs, self._device))
                output = self._predictor.predict(**tensors)
            else:
                output = self._predictor.predict(input_tensor)
            primary = extract_prediction_tensor(output)
            return _extract_model_output(primary).to(device=residual.device, dtype=residual.dtype)

        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"GPU out of memory during inference on device {self._device}. "
                f"Residual shape: {tuple(residual.shape)}. "
                "Try reducing batch size or using CPU."
            ) from e

        except RuntimeError as e:
            if "device" in str(e).lower():
                raise RuntimeError(
                    f"Device mismatch: model on {self._device}, "
                    f"but tensor operation failed. Error: {e}"
                ) from e
            raise RuntimeError(
                f"Neural inference failed: {e}. "
                f"Model device: {self._device}, Residual shape: {tuple(residual.shape)}"
            ) from e

        except (TypeError, AttributeError, ValueError, DLKitError) as e:
            raise RuntimeError(
                f"Unexpected error during neural inference: {type(e).__name__}: {e}"
            ) from e

    def cleanup(self) -> None:
        """Release the underlying DLKit predictor exactly once."""
        if self._closed:
            return
        self._closed = True
        unload = getattr(self._predictor, "unload", None)
        if callable(unload):
            unload()
            return
        self._predictor.__exit__(None, None, None)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Delegate lifecycle cleanup to the idempotent cleanup hook."""
        del exc_type, exc_val, exc_tb
        self.cleanup()


class DLKitAdapter(PredictorAdapter):
    """Adapter for DLKit framework.

    Converts DLKit's API to framework-agnostic PredictorPort interface.
    Handles model loading, device management, and resource cleanup.

    Example:
        >>> adapter = DLKitAdapter()
        >>> with adapter.create_predictor(Path("model.ckpt")) as pred:
        ...     result = pred.apply(residual)
        ...     # Automatic GPU cleanup on exit
    """

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create predictor from DLKit checkpoint.

        Args:
            checkpoint_path: Path to DLKit checkpoint
            config_path: Unused (kept for API compatibility)
            data_config_path: Unused (kept for API compatibility)

        Returns:
            DLKitPredictor instance (use as context manager)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Ensure the path is correct and the file exists."
            )

        try:
            logger.debug(f"Loading checkpoint from: {checkpoint_path}")
            dlkit_predictor = load_model(
                checkpoint_path,
                device="auto",
                # Neural preconditioners must honor fitted checkpoint transforms.
                # PCA-trained models, for example, expect DLKit to project the
                # raw residual into the learned feature space before forward().
                apply_transforms=True,
                precision=PrecisionStrategy.FULL_64,
            )

            model = dlkit_predictor.model
            if model is None:
                raise RuntimeError(
                    f"Failed to load model from checkpoint: {checkpoint_path}. "
                    "Checkpoint may be corrupted or incompatible."
                )

            device_param = next(model.parameters(), None)
            device = str(device_param.device) if device_param is not None else "cpu"
            if device_param is None:
                logger.warning("Model has no parameters, defaulting to CPU")

            logger.info(f"Loaded model from {checkpoint_path} on device {device}")
            # ponytail: parse required_inputs from config_path TOML when DLKit config schema known
            return DLKitPredictor(dlkit_predictor, device, required_inputs=())

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {checkpoint_path}: {type(e).__name__}: {e}"
            ) from e
