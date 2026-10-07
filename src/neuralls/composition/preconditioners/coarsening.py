"""Coarsening-strategy construction for AMG preconditioners.

Builds the `CoarseningStrategy` an AMG preconditioner's hierarchy is built from:
target-dimension search, POD-2G (fit inline or reconstructed from a checkpoint),
neural POD-2G (fit on a neural model's predictions), or aggregation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from dlkit.engine.inference.model_builder import build_model_from_checkpoint
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)

from neuralls.application.inference.prediction import collect_predictions, stack_predictions
from neuralls.composition.preconditioners._weighting import resolve_row_scales
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.dlkit.inference_adapter import create_inference_predictor
from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_parameter_arrays,
)

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy

    from neuralls.domain.inference_ports import InferencePredictorPort

type PODStrategy = PODCoarseningStrategy | SparsePODCoarseningStrategy

PREDICTION_BATCH_SIZE = 256
"""Rows per predictor call when collecting POD snapshots from a neural model."""

type PredictorFactory = Callable[[Path], InferencePredictorPort]
"""Opens an inference predictor for a checkpoint; the default is `create_inference_predictor`."""


def _load_fitted_pod_coarsening(
    checkpoint_path: Path,
    matrix: torch.Tensor,
    pod_cls: type[PODStrategy],
) -> PODStrategy:
    """Reconstruct a fitted POD-2G coarsening strategy from its checkpoint.

    Loads the raw checkpoint dict and rebuilds the module via dlkit's
    generic, trainer-agnostic checkpoint reconstruction
    (`build_model_from_checkpoint`) — the bare `nn.Module`, not the
    `CheckpointPredictor` inference wrapper `dlkit.load_model` returns: a
    coarsening strategy is invoked directly (`build_transfer(A)`) once at
    AMG setup, not through `PredictorPort`'s per-iteration `apply()`.

    Args:
        checkpoint_path: Local path to the fitted `.ckpt` file, already
            resolved/downloaded by
            `composition/assignments/model_resolution.py`.
        matrix: System matrix; used only to match dtype/device.
        pod_cls: Strategy class the checkpoint must reconstruct: the dense
            class for a dense matrix, the sparse class for a CSR matrix.

    Returns:
        A `pod_cls` instance with its `_basis` buffer already loaded from the
        checkpoint (no `.fit()` call needed).

    Raises:
        TypeError: If the reconstructed model is not a `pod_cls`, e.g. a
            checkpoint fitted for the dense class used with a CSR matrix.
    """
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(raw_checkpoint)
    if not isinstance(model, pod_cls):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} reconstructed a {type(model).__name__}, "
            f"expected {pod_cls.__name__}."
        )
    return model.to(dtype=matrix.dtype, device=matrix.device)


def _target_dim_coarsening(cfg: TargetDimCoarseningConfig) -> CoarseningStrategy:
    """Target-dimension coarsening: the candidate search is cached on the strategy."""
    return TargetDimensionCoarsening(
        target_coarse_dim=cfg.target_coarse_dim,
        theta_min=cfg.theta_min,
        theta_max=cfg.theta_max,
        step=cfg.step,
        omega=cfg.omega,
        cache_candidates=True,
    )


def _pod_coarsening(
    cfg: PODCoarseningConfig,
    matrix: torch.Tensor,
    pod_cls: type[PODStrategy],
) -> PODStrategy:
    """POD-2G basis: reconstructed from a fitted checkpoint, or fit inline from snapshots.

    Snapshots are solution vectors, which are dense by nature; only the matrix A is
    sparse, and `build_transfer(A)` never densifies it.
    """
    ckpt = cfg.active_checkpoint_path
    if ckpt is not None:
        # Fitted ahead of time by a `FitJobConfig` assignment (kind dispatch in
        # composition/assignments/comparison_batch.py); the basis is loaded, not refit.
        return _load_fitted_pod_coarsening(ckpt, matrix, pod_cls)

    _, solutions = load_dense_training_arrays(cfg.dataset_dir)
    if cfg.n_snapshots != -1:
        solutions = solutions[: cfg.n_snapshots]
    snapshots = torch.as_tensor(solutions, dtype=matrix.dtype, device=matrix.device)
    row_scales = resolve_row_scales(cfg.weighting, snapshots, matrix)
    coarsening = pod_cls(rank=cfg.rank)
    coarsening.fit(snapshots, row_scales=row_scales)
    return coarsening


def _neural_pod_coarsening(
    cfg: NeuralPODCoarseningConfig,
    matrix: torch.Tensor,
    inference_predictor_factory: PredictorFactory | None,
    pod_cls: type[PODStrategy],
) -> PODStrategy:
    """POD-2G basis fit on a neural model's predictions over the parameter samples."""
    ckpt = cfg.active_checkpoint_path
    if ckpt is None:
        raise ValueError(
            "NeuralPODCoarseningConfig requires checkpoint_path or resolved_checkpoint_path"
        )
    factory = inference_predictor_factory or create_inference_predictor

    param_arrays = load_parameter_arrays(cfg.dataset_dir)
    input_names = cfg.input_names
    if len(input_names) != len(param_arrays):
        raise ValueError(
            f"NeuralPODCoarseningConfig.input_names has {len(input_names)} name(s) but "
            f"dataset_dir={cfg.dataset_dir!r} has {len(param_arrays)} `params` "
            "array(s) — one name per array, in matching order, is required."
        )
    feature_batch = dict(zip(input_names, param_arrays))
    with factory(ckpt) as predictor:
        raw_predictions, _ = collect_predictions(
            predictor, feature_batch, batch_size=PREDICTION_BATCH_SIZE
        )
    predicted = stack_predictions(raw_predictions)
    if cfg.n_snapshots != -1:
        predicted = predicted[: cfg.n_snapshots]
    coarsening = pod_cls(rank=cfg.rank)
    coarsening.fit(torch.as_tensor(predicted, dtype=matrix.dtype, device=matrix.device))
    return coarsening


def _build_amg_coarsening(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: PredictorFactory | None = None,
    *,
    sparse: bool = False,
) -> CoarseningStrategy:
    """Build the coarsening strategy selected by an AMG config's ``coarsening`` field.

    Args:
        matrix: System matrix A; used to match dtype/device of fitted snapshots.
        config: AMG preconditioner configuration.
        inference_predictor_factory: Optional batch-inference predictor factory for
            neural POD-2G coarsening (DI for testing); defaults to
            `create_inference_predictor`.
        sparse: True when A is CSR. POD-2G then uses torchalg's sparse strategy, so
            the transfer operator is built from A without densifying it.

    Returns:
        A ready-to-use coarsening strategy (already fit, if applicable).

    Raises:
        TypeError: For a coarsening with no counterpart on the requested storage:
            target-dimension and aggregation coarsening are dense-only here.
    """
    pod_cls: type[PODStrategy] = SparsePODCoarseningStrategy if sparse else PODCoarseningStrategy
    match config.coarsening:
        case TargetDimCoarseningConfig() as target_dim:
            if sparse:
                raise TypeError("Sparse AMG has no target-dimension coarsening.")
            return _target_dim_coarsening(target_dim)
        case PODCoarseningConfig() as pod:
            return _pod_coarsening(pod, matrix, pod_cls)
        case NeuralPODCoarseningConfig() as neural_pod:
            return _neural_pod_coarsening(neural_pod, matrix, inference_predictor_factory, pod_cls)
        case AggregationCoarseningConfig() as aggregation:
            if sparse:
                raise TypeError("Sparse aggregation AMG is built by vcycle_amg, not here.")
            return AggregationCoarsening(theta=aggregation.theta, omega=aggregation.omega)
        case unknown:
            raise TypeError(f"Unsupported AMG coarsening config: {type(unknown).__name__}")
