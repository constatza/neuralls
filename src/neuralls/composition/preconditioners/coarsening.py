"""Coarsening-strategy construction for AMG preconditioners.

Builds the `CoarseningStrategy` an AMG preconditioner's hierarchy is built from:
target-dimension search, POD (from the basis its method fitted before the timed
setups), neural POD (fit on a neural model's predictions), or aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.utils.pod_basis import compute_pod_basis

from neuralls.application.inference.prediction import collect_predictions, stack_predictions
from neuralls.composition.preconditioners.checkpoint_store import (
    NO_PRELOADED,
    CheckpointLoaders,
    PreloadedCheckpoints,
    preloaded_inference_factory,
    require_loaders,
)
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy


PREDICTION_BATCH_SIZE = 256
"""Rows per predictor call when collecting POD snapshots from a neural model."""


class MissingPODBasisError(ValueError):
    """A POD build was asked for without the basis its method fits before the timed setups."""

    def __init__(self, cfg: PODCoarseningConfig) -> None:
        self.dataset_dir = cfg.dataset_dir
        super().__init__(
            f"No fitted POD basis for dataset_dir={cfg.dataset_dir!r}; the comparison fits it "
            "once per method (composition/comparison/_method_fit.py) before the timed setups"
        )


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
    cfg: PODCoarseningConfig, matrix: torch.Tensor, pod_basis: torch.Tensor | None
) -> PODCoarseningStrategy:
    """POD coarsening from the basis its method fitted once, before the timed setups.

    Only the transfer and coarse matrix are built here, so the timed setup never pays for
    the fit. The basis is dense; `build_transfer(A)` never densifies a CSR A.
    """
    if pod_basis is None:
        raise MissingPODBasisError(cfg)
    return PODCoarseningStrategy(pod_basis.to(dtype=matrix.dtype, device=matrix.device))


def _neural_pod_coarsening(
    cfg: NeuralPODCoarseningConfig,
    matrix: torch.Tensor,
    loaders: CheckpointLoaders | None,
    preloaded: PreloadedCheckpoints,
    staged_fit_data: tuple[np.ndarray, ...] | None = None,
) -> PODCoarseningStrategy:
    """POD basis fit on a neural model's predictions over the parameter samples."""
    ckpt = cfg.active_checkpoint_path
    if ckpt is None:
        raise ValueError(
            "NeuralPODCoarseningConfig requires checkpoint_path or resolved_checkpoint_path"
        )
    # Loaders are needed even when the predictor was preloaded: their device is the one the
    # prediction's memory is measured on.
    loaders = require_loaders(loaders, "neural POD coarsening")
    factory = preloaded_inference_factory(loaders.inference_predictor_factory, preloaded)

    if staged_fit_data is None:
        raise ValueError(
            f"Parameter arrays for dataset_dir={cfg.dataset_dir!r} were not staged; "
            "PreconditionerService.preload stages them before the build is timed"
        )
    param_arrays = staged_fit_data
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
            predictor, feature_batch, batch_size=PREDICTION_BATCH_SIZE, device=loaders.device
        )
    predicted = stack_predictions(raw_predictions)
    if cfg.n_snapshots != -1:
        predicted = predicted[: cfg.n_snapshots]
    snapshots = torch.as_tensor(predicted, dtype=matrix.dtype, device=matrix.device)
    return PODCoarseningStrategy(compute_pod_basis(snapshots, cfg.rank))


def _build_amg_coarsening(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    loaders: CheckpointLoaders | None = None,
    *,
    sparse: bool = False,
    preloaded: PreloadedCheckpoints = NO_PRELOADED,
    staged_fit_data: tuple[np.ndarray, ...] | None = None,
    pod_basis: torch.Tensor | None = None,
) -> CoarseningStrategy:
    """Build the coarsening strategy selected by an AMG config's ``coarsening`` field.

    Args:
        matrix: System matrix A; used to match dtype/device of fitted snapshots.
        config: AMG preconditioner configuration.
        loaders: Checkpoint loaders and their device, needed only by neural POD
            coarsening.
        sparse: True when A is CSR; target-dimension and aggregation coarsening
            are then rejected. POD serves both layouts and never densifies A.
        preloaded: Checkpoints loaded ahead of the build (see `checkpoint_store`);
            the default loads inside the build.
        staged_fit_data: Parameter arrays a neural POD predicts its snapshots from.
        pod_basis: The basis a POD method fitted before its timed setups.

    Returns:
        A ready-to-use coarsening strategy (already fit, if applicable).

    Raises:
        TypeError: For a coarsening with no counterpart on the requested storage:
            target-dimension and aggregation coarsening are dense-only here.
    """
    match config.coarsening:
        case TargetDimCoarseningConfig() as target_dim:
            if sparse:
                raise TypeError("Sparse AMG has no target-dimension coarsening.")
            return _target_dim_coarsening(target_dim)
        case PODCoarseningConfig() as pod:
            return _pod_coarsening(pod, matrix, pod_basis)
        case NeuralPODCoarseningConfig() as neural_pod:
            return _neural_pod_coarsening(
                neural_pod,
                matrix,
                loaders,
                preloaded,
                staged_fit_data,
            )
        case AggregationCoarseningConfig() as aggregation:
            if sparse:
                raise TypeError("Sparse aggregation AMG is built by vcycle_amg, not here.")
            return AggregationCoarsening(theta=aggregation.theta, omega=aggregation.omega)
        case unknown:
            raise TypeError(f"Unsupported AMG coarsening config: {type(unknown).__name__}")
