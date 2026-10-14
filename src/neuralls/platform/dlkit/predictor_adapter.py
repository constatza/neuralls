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
from torchalg.preconditioners.implementations import NeuralPreconditioner
from torchalg.preconditioners.ports import (
    ExtraInputPredictorPort,
    PredictorAdapter,
)

from ._prediction_outputs import extract_prediction_tensor

UNKNOWN_MODEL_CLASS_NAME = "unknown"
"""Placeholder when a predictor is built without a model to name (never from `DLKitAdapter`)."""

if TYPE_CHECKING:
    from dlkit.interfaces.inference import CheckpointPredictor as DLKitCheckpointPredictor


def _extra_to_tensors(
    extra_inputs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Move extra input tensors to the predictor device and add a batch axis."""
    return {name: _prepare_model_input(arr, device) for name, arr in extra_inputs.items()}


# TODO: float32 is not supported anywhere yet; float64 is hardcoded in this module
# (`_prepare_model_input`, `_extract_model_output`, and `PrecisionStrategy.FULL_64` in
# `DLKitAdapter.create_predictor`). Make the precision a preconditioner-level setting
# (a field on `NeuralPreconditionerConfig`, part of its identity) and derive the cast
# dtype and the `load_model` precision from it. Until then a float32 solve silently
# casts every extra input, including the full matrix, to float64 on every
# `apply()`, and fp64 is slow on most GPUs.
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
        model_class_name: str = UNKNOWN_MODEL_CLASS_NAME,
    ) -> None:
        """Initialize predictor with loaded CheckpointPredictor.

        Args:
            predictor: DLKit CheckpointPredictor (already loaded, eval mode)
            device: Device string ("cpu", "cuda", "mps")
            required_inputs: Names of extra arrays the model expects beyond the residual.
                Derived from the model config by the adapter; eliminates TOML duplication.
            model_class_name: Architecture class name of the loaded model, recorded by the
                adapter that loaded it (which holds the model) for display labels.
        """
        self._predictor: DLKitCheckpointPredictor = predictor
        self._model_class_name: str = model_class_name
        self._device: str = device
        self._required_inputs: tuple[str, ...] = required_inputs
        self._closed = False

    @property
    def model_class_name(self) -> str:
        """Class name of the loaded torch model (its architecture), e.g. ``"GraphNet"``.

        Captured at load time so it stays available after ``cleanup()`` unloads the model.
        """
        return self._model_class_name

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Names of extra arrays this model expects beyond the residual."""
        # ponytail: parse from DLKit config_path when the config schema is known;
        # currently populated by DLKitAdapter from config if available, else empty.
        return self._required_inputs

    def check_extra_inputs(self, extra_input_names: tuple[str, ...]) -> None:
        """Fail fast if declared extra inputs do not fit the checkpoint's features.

        Every declared name must be a feature of the checkpoint, and exactly one
        feature must be left over for the residual (the same resolution
        `apply()` performs on every call, here done once up front so a
        misconfigured ``extra_input_names`` fails when the preconditioner is built
        instead of mid-solve). Checkpoints that persisted no feature names, and an
        empty declaration (which `apply()` serves with a positional call), are not
        checked.

        Args:
            extra_input_names: Names the config declares as extra model inputs.

        Raises:
            ValueError: If a name is not a checkpoint feature, or the residual's
                feature cannot be identified uniquely.
        """
        if not extra_input_names:
            return
        feature_names = self._predictor.feature_names
        if not feature_names:
            return
        unknown = sorted(set(extra_input_names) - set(feature_names))
        if unknown:
            raise ValueError(
                f"extra_input_names {unknown} are not features of the checkpoint "
                f"(feature_names={feature_names!r}). Check extra_input_names against the "
                "checkpoint's actual forward() parameters."
            )
        _resolve_primary_input_name(feature_names, frozenset(extra_input_names))

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


class DLKitNeuralPreconditioner(NeuralPreconditioner):
    """`NeuralPreconditioner` that can name the architecture it loaded, for display labels.

    torchalg keeps the predictor private, so a label has no public way to ask
    which model is behind a neural preconditioner. This subclass is the
    sanctioned extension seam: it reads the (protected) predictor and exposes
    only the class name.

    It *borrows* its predictor, like every torchalg `NeuralPreconditioner`: the
    caller (the checkpoint store, or `OwningDLKitNeuralPreconditioner`) releases
    it, so one loaded predictor can back several preconditioners.

    A DLKit predictor's declared ``extra_input_names`` are checked against the
    checkpoint's features at construction (`DLKitPredictor.check_extra_inputs`).
    """

    def __init__(
        self, predictor: ExtraInputPredictorPort, extra_input_names: tuple[str, ...] = ()
    ) -> None:
        if isinstance(predictor, DLKitPredictor):
            predictor.check_extra_inputs(extra_input_names)
        super().__init__(predictor, extra_input_names)

    @property
    def model_class_name(self) -> str:
        """Architecture class name; valid once ``setup()`` has loaded the predictor."""
        predictor = self._predictor
        if isinstance(predictor, DLKitPredictor):
            return predictor.model_class_name
        return type(predictor).__name__


class OwningDLKitNeuralPreconditioner(DLKitNeuralPreconditioner):
    """A `DLKitNeuralPreconditioner` that releases the predictor it was handed.

    For a predictor loaded just for this preconditioner and shared with nobody:
    the GPU memory is freed with the preconditioner, explicitly through
    ``cleanup()`` or, failing that, when it is garbage collected. A predictor
    that outlives the preconditioner (preloaded and shared) must go to
    `DLKitNeuralPreconditioner` instead, whose owner releases it.
    """

    def cleanup(self) -> None:
        """Release the owned predictor (idempotent)."""
        self._predictor.cleanup()

    def __del__(self) -> None:
        """Release the owned predictor when garbage collected."""
        predictor = getattr(self, "_predictor", None)
        if predictor is not None:
            predictor.cleanup()


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
                # TODO: float32 unsupported; see the module-level precision TODO above.
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
            return DLKitPredictor(
                dlkit_predictor,
                device,
                required_inputs=(),
                model_class_name=type(model).__name__,
            )

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {checkpoint_path}: {type(e).__name__}: {e}"
            ) from e
