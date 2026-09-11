"""Preconditioner factory for TOML workflow.

This module provides a simple factory function for creating preconditioners
from configuration objects. It lives in the assembly layer because it depends
on both the configuration layer (PreconditionerType, config models) and the
solver layer (concrete preconditioner classes).

For direct usage, just instantiate preconditioners directly:
    >>> precond = JacobiPreconditioner(matrix)

For TOML workflow:
    >>> config = load_comparison_config("comparison.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)

Design:
    - Simple factory function with isinstance checks and explicit mapping
    - No builder classes needed - just call preconditioner constructors
    - Supports dependency injection for neural preconditioners (testing)
    - Better type safety via isinstance checks (understood by mypy)
    - Explicit enum-to-class mapping for clarity and extensibility
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    IC0Preconditioner,
    ICholeskyPreconditioner,
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
)

from neuralls.platform.config.models.preconditioner import PreconditionerType

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
    from torchalg.preconditioners.ports import PredictorAdapter

    from neuralls.domain.inference_ports import InferencePredictorPort
    from neuralls.platform.config.models.preconditioner import (
        AMGPreconditionerConfig,
        ConcretePreconditionerConfig,
    )


@dataclass(frozen=True)
class AMGBuild:
    """An assembled `AMGPreconditioner` plus the coarsening strategy that built it.

    `AMGPreconditioner` keeps its `coarsening` as a private attribute (it is
    runtime state, not part of its public API), so diagnostics code that
    needs the coarsening strategy itself (e.g. to read back the realized
    coarse dimension via its public `build_transfer`) cannot get it from the
    preconditioner after the fact without reaching into private state.
    Returning both from the same build call means diagnostics reuse the
    exact coarsening object already used to build the hierarchy — no
    duplicate POD fit, no duplicate checkpoint load.

    Attributes:
        preconditioner: The assembled `AMGPreconditioner`.
        coarsening: The coarsening strategy used to build it.
    """

    preconditioner: Preconditioner
    coarsening: CoarseningStrategy


@dataclass(frozen=True)
class PreconditionerScheduleConfig:
    """Scheduling parameters for preconditioner switching.

    Extracted from BasePreconditionerConfig for internal use.
    Separates scheduling concerns from preconditioner configuration.

    Attributes:
        start_iter: Iteration at which the primary preconditioner becomes active.
        limit_iters: Number of iterations to apply primary preconditioner.
                     -1 means unlimited (use primary for entire solve).
        fallback: Preconditioner type to switch to after limit is reached.
    """

    start_iter: int = 0
    limit_iters: int = -1
    fallback: PreconditionerType = PreconditionerType.IDENTITY


def _load_fitted_pod_coarsening(
    checkpoint_path: Path, matrix: torch.Tensor
) -> PODCoarseningStrategy:
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

    Returns:
        A `PODCoarseningStrategy` with its `_basis` buffer already loaded
        from the checkpoint (no `.fit()` call needed).

    Raises:
        TypeError: If the reconstructed model is not a `PODCoarseningStrategy`.
    """
    from dlkit.engine.inference.model_builder import build_model_from_checkpoint
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(raw_checkpoint)
    if not isinstance(model, PODCoarseningStrategy):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} reconstructed a {type(model).__name__}, "
            "expected PODCoarseningStrategy."
        )
    return model.to(dtype=matrix.dtype, device=matrix.device)


def _build_amg_coarsening(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: Callable[[Path, Any], InferencePredictorPort] | None = None,
) -> CoarseningStrategy:
    """Build the coarsening strategy selected by an AMG config's ``coarsening`` field.

    Args:
        matrix: System matrix A; used to match dtype/device of fitted snapshots.
        config: AMG preconditioner configuration.
        inference_predictor_factory: Optional batch-inference predictor factory
            for neural POD-2G coarsening (DI for testing); defaults to
            `create_inference_predictor` from `platform.dlkit.inference_adapter`.

    Returns:
        A ready-to-use coarsening strategy (already fit, if applicable).
    """
    from torchalg.preconditioners.implementations.amg import AggregationCoarsening

    from neuralls.platform.config.models.preconditioner import (
        NeuralPODCoarseningConfig,
        PODCoarseningConfig,
        TargetDimCoarseningConfig,
    )

    if isinstance(config.coarsening, TargetDimCoarseningConfig):
        from torchalg.preconditioners.implementations.amg import TargetDimensionCoarsening

        td_cfg = config.coarsening
        return TargetDimensionCoarsening(
            target_coarse_dim=td_cfg.target_coarse_dim,
            theta_min=td_cfg.theta_min,
            theta_max=td_cfg.theta_max,
            step=td_cfg.step,
            omega=td_cfg.omega,
            cache_candidates=True,
        )

    if isinstance(config.coarsening, PODCoarseningConfig):
        pod_cfg = config.coarsening
        ckpt = pod_cfg.resolved_checkpoint_path or pod_cfg.checkpoint_path
        if ckpt is not None:
            # Fitted ahead of time via a `FitJobConfig` assignment
            # (composition/assignments/comparison_batch.py's kind
            # dispatch) — reconstruct the fitted basis directly from its
            # MLflow-tracked checkpoint instead of refitting inline.
            return _load_fitted_pod_coarsening(ckpt, matrix)

        # Not yet migrated to an assignment: fit inline from raw
        # snapshot files, exactly as before (backward compatible).
        from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

        from neuralls.platform.storage.dataset_readers import load_dense_training_arrays

        _, solutions = load_dense_training_arrays(pod_cfg.dataset_dir)
        if pod_cfg.n_snapshots != -1:
            solutions = solutions[: pod_cfg.n_snapshots]
        coarsening = PODCoarseningStrategy(rank=pod_cfg.rank)
        coarsening.fit(torch.as_tensor(solutions, dtype=matrix.dtype, device=matrix.device))
        return coarsening

    if isinstance(config.coarsening, NeuralPODCoarseningConfig):
        from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

        from neuralls.application.inference.prediction import (
            collect_predictions,
            stack_predictions,
        )
        from neuralls.platform.storage.dataset_readers import load_parameter_arrays

        neural_pod_cfg = config.coarsening
        ckpt = neural_pod_cfg.resolved_checkpoint_path or neural_pod_cfg.checkpoint_path
        if ckpt is None:
            raise ValueError(
                "NeuralPODCoarseningConfig requires checkpoint_path or resolved_checkpoint_path"
            )
        if inference_predictor_factory is None:
            from neuralls.platform.dlkit.inference_adapter import create_inference_predictor

            inference_predictor_factory = create_inference_predictor

        param_arrays = load_parameter_arrays(neural_pod_cfg.dataset_dir)
        input_names = neural_pod_cfg.input_names
        if len(input_names) != len(param_arrays):
            raise ValueError(
                f"NeuralPODCoarseningConfig.input_names has {len(input_names)} name(s) but "
                f"dataset_dir={neural_pod_cfg.dataset_dir!r} has {len(param_arrays)} `params` "
                "array(s) — one name per array, in matching order, is required."
            )
        feature_batch = dict(zip(input_names, param_arrays))
        with inference_predictor_factory(ckpt, None) as predictor:
            raw_predictions, _ = collect_predictions(predictor, feature_batch, batch_size=256)
        predicted = stack_predictions(raw_predictions)
        if neural_pod_cfg.n_snapshots != -1:
            predicted = predicted[: neural_pod_cfg.n_snapshots]
        coarsening = PODCoarseningStrategy(rank=neural_pod_cfg.rank)
        coarsening.fit(torch.as_tensor(predicted, dtype=matrix.dtype, device=matrix.device))
        return coarsening

    return AggregationCoarsening(theta=config.coarsening.theta, omega=config.coarsening.omega)


def _build_amg(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: Callable[[Path, Any], InferencePredictorPort] | None = None,
) -> AMGBuild:
    """Assemble an `AMGPreconditioner` and return it alongside its coarsening strategy.

    Args:
        matrix: System matrix A.
        config: AMG preconditioner configuration.
        inference_predictor_factory: Optional batch-inference predictor factory
            for neural POD-2G coarsening (DI for testing).

    Returns:
        The assembled preconditioner and the coarsening strategy used to build it.
    """
    from torchalg.preconditioners.implementations.amg import (
        AMGPreconditioner,
        JacobiSmoother,
        VCycle,
    )

    coarsening = _build_amg_coarsening(matrix, config, inference_predictor_factory)
    smoother = JacobiSmoother(omega=config.smoother_omega)
    cycle = VCycle(
        smoother=smoother,
        n_pre=config.pre_smoothing_steps,
        n_post=config.post_smoothing_steps,
    )
    preconditioner = AMGPreconditioner(
        matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=True
    )
    return AMGBuild(preconditioner=preconditioner, coarsening=coarsening)


def create_preconditioner(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: Callable[[Path, Any], InferencePredictorPort] | None = None,
) -> Preconditioner:
    """Create preconditioner from configuration.

    Uses isinstance checks for type narrowing and explicit mapping
    for better type safety and IDE support.

    Args:
        matrix: System matrix A
        config: Preconditioner configuration from TOML
        adapter: Optional adapter for neural preconditioner (DI for testing)
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing); defaults
            to `create_inference_predictor` from `platform.dlkit.inference_adapter`.

    Returns:
        Preconditioner instance

    Example:
        >>> # Load from TOML
        >>> config = load_comparison_config("comparison.toml")
        >>> precond = create_preconditioner(A, config.preconditioner)
        >>>
        >>> # Use it
        >>> z = precond.apply(residual)

    Raises:
        ValueError: If preconditioner type is not supported
    """
    from neuralls.platform.config.models.preconditioner import (
        AMGPreconditionerConfig,
        IC0PreconditionerConfig,
        NeuralAMGPreconditionerConfig,
        NeuralPreconditionerConfig,
    )

    # AMG family: config.coarsening selects the strategy that builds the
    # prolongation/restriction operator (aggregation vs. POD-2G); everything
    # else (cycle, smoothing, n_levels) is shared.
    if config.type == PreconditionerType.AMG:
        if not isinstance(config, AMGPreconditionerConfig):
            raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
        return _build_amg(matrix, config, inference_predictor_factory).preconditioner

    # Neural AMG (neural prolongation/restriction, stub)
    if config.type == PreconditionerType.NEURAL_AMG:
        if not isinstance(config, NeuralAMGPreconditionerConfig):
            raise TypeError(
                f"NEURAL_AMG type requires NeuralAMGPreconditionerConfig, got {type(config)}"
            )
        if adapter is None:
            from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

            adapter = DLKitAdapter()
        p_cfg = config.prolongation
        ckpt_p = p_cfg.resolved_checkpoint_path or p_cfg.checkpoint_path
        if ckpt_p is None:
            raise ValueError("NeuralAMGPreconditionerConfig.prolongation requires a checkpoint")
        prolongator = adapter.create_predictor(ckpt_p, p_cfg.config_path, p_cfg.data_config_path)
        restrictor = None
        if config.restriction is not None:
            r_cfg = config.restriction
            ckpt_r = r_cfg.resolved_checkpoint_path or r_cfg.checkpoint_path
            if ckpt_r is None:
                raise ValueError("NeuralAMGPreconditionerConfig.restriction requires a checkpoint")
            restrictor = adapter.create_predictor(ckpt_r, r_cfg.config_path, r_cfg.data_config_path)
        from torchalg.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            NeuralCoarseningStrategy,
            VCycle,
        )

        coarsening = NeuralCoarseningStrategy(prolongator=prolongator, restrictor=restrictor)
        smoother = JacobiSmoother(omega=config.smoother_omega)
        cycle = VCycle(
            smoother=smoother,
            n_pre=config.pre_smoothing_steps,
            n_post=config.post_smoothing_steps,
        )
        return AMGPreconditioner(
            matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=False
        )

    # Check if type is NEURAL but config is not NeuralPreconditionerConfig
    if config.type == PreconditionerType.NEURAL:
        if not isinstance(config, NeuralPreconditionerConfig):
            raise TypeError(f"Neural type requires NeuralPreconditionerConfig, got {type(config)}")
        ckpt = config.resolved_checkpoint_path or config.checkpoint_path
        if ckpt is None:
            raise ValueError(
                "NeuralPreconditionerConfig requires checkpoint_path or resolved_checkpoint_path"
            )
        if adapter is None:
            from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

            adapter = DLKitAdapter()
        return NeuralPreconditioner(
            checkpoint_path=ckpt,
            config_path=config.config_path,
            data_config_path=config.data_config_path,
            adapter=adapter,
            extra_input_names=tuple(config.extra_input_names),
        )

    # IC(0) with threshold parameter
    if config.type == PreconditionerType.IC0:
        if not isinstance(config, IC0PreconditionerConfig):
            raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
        return IC0Preconditioner(matrix, threshold=config.threshold)

    # Standard cases with explicit dispatch for type safety
    if config.type in (PreconditionerType.IDENTITY, PreconditionerType.NONE):
        return Identity()
    if config.type == PreconditionerType.JACOBI:
        return JacobiPreconditioner(matrix)
    if config.type == PreconditionerType.ILU:
        return ILUPreconditioner(matrix)
    if config.type == PreconditionerType.ICHOLESKY:
        return ICholeskyPreconditioner(matrix)

    raise ValueError(f"Unsupported preconditioner type: {config.type}")


def create_preconditioner_with_coarsening(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: Callable[[Path, Any], InferencePredictorPort] | None = None,
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Create a preconditioner, also returning its coarsening strategy when it has one.

    AMG's realized coarse dimension is only knowable from the coarsening
    strategy actually used to build the hierarchy, not from config alone
    (POD's ``rank`` is often an energy threshold; AMG's ``theta`` yields an
    emergent aggregate count). Diagnostics that need the realized dimension
    must reuse this coarsening object rather than fitting a second one.

    Args:
        matrix: System matrix A.
        config: Preconditioner configuration from TOML.
        adapter: Optional adapter for neural preconditioner (DI for testing).
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing).

    Returns:
        The preconditioner, and its coarsening strategy if `config.type` is
        `PreconditionerType.AMG` (`None` for every other preconditioner type).
    """
    from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig

    if config.type == PreconditionerType.AMG:
        if not isinstance(config, AMGPreconditionerConfig):
            raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
        build = _build_amg(matrix, config, inference_predictor_factory)
        return build.preconditioner, build.coarsening

    return create_preconditioner(matrix, config, adapter, inference_predictor_factory), None


def _extract_schedule(cfg: ConcretePreconditionerConfig) -> PreconditionerScheduleConfig:
    """Extract scheduling parameters from preconditioner config.

    Pure function to extract scheduling concerns from mixed config.

    Args:
        cfg: Preconditioner configuration from TOML

    Returns:
        Extracted schedule configuration
    """
    return PreconditionerScheduleConfig(
        start_iter=cfg.start_iter,
        limit_iters=cfg.limit_iters,
        fallback=cfg.fallback,
    )


def create_scheduled_preconditioner(
    primary: Preconditioner,
    schedule: PreconditionerScheduleConfig,
) -> Preconditioner:
    """Create a scheduled preconditioner based on schedule config.

    Args:
        primary: Main preconditioner to apply
        schedule: Schedule configuration with activation, limit, and fallback type

    Returns:
        ScheduledPreconditioner if delayed or limited, otherwise primary unchanged

    Example:
        >>> # Limit neural preconditioner to first 10 iterations
        >>> schedule = PreconditionerScheduleConfig(limit_iters=10)
        >>> scheduled = create_scheduled_preconditioner(neural_precond, schedule)
    """
    if schedule.start_iter == 0 and schedule.limit_iters < 0:
        return primary

    from torchalg.preconditioners.implementations.scheduled import (
        ScheduledPreconditioner,
    )

    # Create fallback preconditioner based on type
    if schedule.fallback == PreconditionerType.IDENTITY:
        fallback_precond = Identity()
    else:
        raise ValueError(f"Unsupported fallback type: {schedule.fallback}")

    return ScheduledPreconditioner(
        primary=primary,
        fallback=fallback_precond,
        limit_iters=None if schedule.limit_iters < 0 else schedule.limit_iters,
        start_iter=schedule.start_iter,
    )
