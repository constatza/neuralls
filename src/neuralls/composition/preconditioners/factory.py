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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from dlkit.engine.inference.model_builder import build_model_from_checkpoint
from loguru import logger
from torchalg.multigrid import AMGPreconditioner as MultigridAMGPreconditioner
from torchalg.multigrid import VCycle as MultigridVCycle
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    IC0Preconditioner,
    ICholeskyPreconditioner,
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    AMGPreconditioner,
    JacobiSmoother,
    NeuralCoarseningStrategy,
    TargetDimensionCoarsening,
    VCycle,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_coarse_solve
from torchalg.sparse.preconditioners.amg.smoothers import resolve_jacobi_default
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner as SparseIC0Preconditioner
from torchalg.sparse.preconditioners.icholesky import (
    ICholeskyPreconditioner as SparseICholeskyPreconditioner,
)
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner as SparseILUPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)

from neuralls.application.inference.prediction import collect_predictions, stack_predictions
from neuralls.composition.preconditioners._weighting import resolve_row_scales
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
    TargetDimCoarseningConfig,
)
from neuralls.platform.dlkit.inference_adapter import create_inference_predictor
from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_parameter_arrays,
)
from neuralls.shared.types import MatrixFormat

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
    from torchalg.preconditioners.ports import PredictorAdapter

    from neuralls.domain.inference_ports import InferencePredictorPort
    from neuralls.platform.config.models.preconditioner import ConcretePreconditionerConfig


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


type PODStrategy = PODCoarseningStrategy | SparsePODCoarseningStrategy


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
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None,
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
    with factory(ckpt, None) as predictor:
        raw_predictions, _ = collect_predictions(predictor, feature_batch, batch_size=256)
    predicted = stack_predictions(raw_predictions)
    if cfg.n_snapshots != -1:
        predicted = predicted[: cfg.n_snapshots]
    coarsening = pod_cls(rank=cfg.rank)
    coarsening.fit(torch.as_tensor(predicted, dtype=matrix.dtype, device=matrix.device))
    return coarsening


def _build_amg_coarsening(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None = None,
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
        case aggregation:
            if sparse:
                raise TypeError("Sparse aggregation AMG is built by vcycle_amg, not here.")
            return AggregationCoarsening(theta=aggregation.theta, omega=aggregation.omega)


def _build_amg(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None = None,
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


def _build_adaptive_sa(
    matrix: torch.Tensor,
    config: AdaptiveSAPreconditionerConfig,
) -> Preconditioner:
    """Assemble an `AdaptiveSAPreconditioner` (alpha-SA) at a fixed `theta`.

    No target-dimension search: `AdaptiveSAPreconditioner` builds its whole
    hierarchy eagerly from `theta` and has no pluggable `CoarseningStrategy`
    to hand a target to the way `TargetDimensionCoarsening` does for
    classical SA-AMG, and an end-to-end run showed a from-scratch search
    wrapper isn't worth it anyway — pick `theta` directly instead (see the
    class docstring's measured `theta -> c` table).

    Args:
        matrix: System matrix A.
        config: Adaptive-SA preconditioner configuration.

    Returns:
        The assembled `AdaptiveSAPreconditioner`.
    """
    from torchalg.preconditioners.implementations.amg import AdaptiveSAPreconditioner

    return AdaptiveSAPreconditioner(
        matrix,
        num_candidates=config.num_candidates,
        candidate_iters=config.candidate_iters,
        max_levels=config.n_levels,
        max_coarse=config.max_coarse,
        theta=config.theta,
        omega=config.omega,
        seed=config.seed,
    )


def _build_bootstrap_amg(
    matrix: torch.Tensor,
    config: BootstrapAMGPreconditionerConfig,
) -> Preconditioner:
    """Assemble a `BootstrapAMGPreconditioner` (BAMG).

    Like `_build_adaptive_sa`, its whole hierarchy is built eagerly from the
    config's fields — no pluggable `CoarseningStrategy` to hand a target to.

    Args:
        matrix: System matrix A.
        config: Bootstrap-AMG preconditioner configuration.

    Returns:
        The assembled `BootstrapAMGPreconditioner`.
    """
    from torchalg.preconditioners.implementations.amg import BootstrapAMGPreconditioner

    return BootstrapAMGPreconditioner(
        matrix,
        k_r=config.k_r,
        eta=config.eta,
        n_bootstrap_cycles=config.n_bootstrap_cycles,
        nu=config.nu,
        delta=config.delta,
        theta_ad=config.theta_ad,
        caliber=config.caliber,
        gamma=config.gamma,
        use_lsr=config.use_lsr,
        max_levels=config.n_levels,
        max_coarse=config.max_coarse,
        seed=config.seed,
    )


@dataclass(frozen=True)
class _BuildDeps:
    """Injected collaborators shared by every preconditioner builder."""

    adapter: PredictorAdapter | None
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None


type PreconditionerBuilder = Callable[
    [torch.Tensor, ConcretePreconditionerConfig, _BuildDeps], Preconditioner
]
"""Builds one preconditioner from a matrix already in the builder's format."""


def _build_identity(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Return the no-op preconditioner; it ignores the matrix in either format."""
    del matrix, config, deps
    return Identity()


def _build_dense_jacobi(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense diagonal (Jacobi) preconditioner."""
    del config, deps
    return JacobiPreconditioner(matrix)


def _build_sparse_jacobi(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR diagonal (Jacobi) preconditioner."""
    del config, deps
    return SparseJacobiPreconditioner(matrix)


def _build_dense_ilu(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense ILU(0) preconditioner."""
    del config, deps
    return ILUPreconditioner(matrix)


def _build_sparse_ilu(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR ILU(0) preconditioner."""
    del config, deps
    return SparseILUPreconditioner(matrix)


def _build_dense_icholesky(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense incomplete-Cholesky preconditioner over a supplied factor."""
    del config, deps
    return ICholeskyPreconditioner(matrix)


def _build_sparse_icholesky(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR incomplete-Cholesky preconditioner over a supplied factor."""
    del config, deps
    return SparseICholeskyPreconditioner(matrix)


def _build_dense_ic0(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense IC(0) preconditioner with its drop threshold."""
    del deps
    if not isinstance(config, IC0PreconditionerConfig):
        raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
    return IC0Preconditioner(matrix, threshold=config.threshold)


def _build_sparse_ic0(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR IC(0) preconditioner with its drop threshold."""
    del deps
    if not isinstance(config, IC0PreconditionerConfig):
        raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
    return SparseIC0Preconditioner(matrix, threshold=config.threshold)


def _build_sparse_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR AMG: A stays CSR in every level and in the transfer operators.

    Aggregation coarsening uses torchalg's sparse V-cycle preset. POD-2G and neural
    POD-2G use the same multigrid engine with torchalg's sparse POD strategy, so the
    coarse operator is formed as P.T A P without densifying A. Target-dimension
    coarsening has no sparse counterpart and is rejected.
    """
    if not isinstance(config, AMGPreconditionerConfig):
        raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
    if isinstance(config.coarsening, AggregationCoarseningConfig):
        return sparse_vcycle_amg(
            matrix,
            theta=config.coarsening.theta,
            omega=config.coarsening.omega,
            n_levels=config.n_levels,
            smoother_omega=config.smoother_omega,
            n_pre=config.pre_smoothing_steps,
            n_post=config.post_smoothing_steps,
        )
    coarsening = _build_amg_coarsening(
        matrix, config, deps.inference_predictor_factory, sparse=True
    )
    cycle = MultigridVCycle(
        resolve_jacobi_default(None, config.smoother_omega),
        n_pre=config.pre_smoothing_steps,
        n_post=config.post_smoothing_steps,
        coarse_solver=dense_coarse_solve,
    )
    return MultigridAMGPreconditioner(
        matrix=matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=True
    )


def _build_dense_adaptive_sa(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense alpha-SA AMG preconditioner."""
    del deps
    if not isinstance(config, AdaptiveSAPreconditionerConfig):
        raise TypeError(
            f"ADAPTIVE_SA_AMG type requires AdaptiveSAPreconditionerConfig, got {type(config)}"
        )
    return _build_adaptive_sa(matrix, config)


def _build_sparse_adaptive_sa(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR alpha-SA AMG preconditioner."""
    del deps
    if not isinstance(config, AdaptiveSAPreconditionerConfig):
        raise TypeError(
            f"ADAPTIVE_SA_AMG type requires AdaptiveSAPreconditionerConfig, got {type(config)}"
        )
    return SparseAdaptiveSAPreconditioner(
        matrix,
        num_candidates=config.num_candidates,
        candidate_iters=config.candidate_iters,
        max_levels=config.n_levels,
        max_coarse=config.max_coarse,
        theta=config.theta,
        omega=config.omega,
        seed=config.seed,
    )


def _build_dense_bootstrap_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Dense bootstrap AMG (BAMG) preconditioner."""
    del deps
    if not isinstance(config, BootstrapAMGPreconditionerConfig):
        raise TypeError(
            f"BOOTSTRAP_AMG type requires BootstrapAMGPreconditionerConfig, got {type(config)}"
        )
    return _build_bootstrap_amg(matrix, config)


def _build_sparse_bootstrap_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Sparse CSR bootstrap AMG (BAMG) preconditioner."""
    del deps
    if not isinstance(config, BootstrapAMGPreconditionerConfig):
        raise TypeError(
            f"BOOTSTRAP_AMG type requires BootstrapAMGPreconditionerConfig, got {type(config)}"
        )
    return SparseBootstrapAMGPreconditioner(
        matrix,
        k_r=config.k_r,
        eta=config.eta,
        n_bootstrap_cycles=config.n_bootstrap_cycles,
        nu=config.nu,
        delta=config.delta,
        theta_ad=config.theta_ad,
        caliber=config.caliber,
        gamma=config.gamma,
        use_lsr=config.use_lsr,
        max_levels=config.n_levels,
        max_coarse=config.max_coarse,
        seed=config.seed,
    )


def _build_neural(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Checkpoint-backed neural preconditioner; the matrix is not used, only the model."""
    del matrix
    if not isinstance(config, NeuralPreconditionerConfig):
        raise TypeError(f"Neural type requires NeuralPreconditionerConfig, got {type(config)}")
    ckpt = config.active_checkpoint_path
    if ckpt is None:
        raise ValueError(
            "NeuralPreconditionerConfig requires checkpoint_path or resolved_checkpoint_path"
        )
    return NeuralPreconditioner(
        checkpoint_path=ckpt,
        config_path=config.config_path,
        data_config_path=config.data_config_path,
        adapter=_resolve_adapter(deps),
        extra_input_names=tuple(config.extra_input_names),
    )


def _build_neural_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> Preconditioner:
    """Neural prolongation/restriction AMG preconditioner (dense input only)."""
    if not isinstance(config, NeuralAMGPreconditionerConfig):
        raise TypeError(
            f"NEURAL_AMG type requires NeuralAMGPreconditionerConfig, got {type(config)}"
        )
    adapter = _resolve_adapter(deps)
    p_cfg = config.prolongation
    ckpt_p = p_cfg.active_checkpoint_path
    if ckpt_p is None:
        raise ValueError("NeuralAMGPreconditionerConfig.prolongation requires a checkpoint")
    prolongator = adapter.create_predictor(ckpt_p, p_cfg.config_path, p_cfg.data_config_path)
    restrictor = None
    if config.restriction is not None:
        r_cfg = config.restriction
        ckpt_r = r_cfg.active_checkpoint_path
        if ckpt_r is None:
            raise ValueError("NeuralAMGPreconditionerConfig.restriction requires a checkpoint")
        restrictor = adapter.create_predictor(ckpt_r, r_cfg.config_path, r_cfg.data_config_path)
    coarsening = NeuralCoarseningStrategy(prolongator=prolongator, restrictor=restrictor)
    cycle = VCycle(
        smoother=JacobiSmoother(omega=config.smoother_omega),
        n_pre=config.pre_smoothing_steps,
        n_post=config.post_smoothing_steps,
    )
    return AMGPreconditioner(
        matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=False
    )


def _resolve_adapter(deps: _BuildDeps) -> PredictorAdapter:
    """Return the injected predictor adapter, or the DLKit default when none was injected."""
    from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

    return deps.adapter if deps.adapter is not None else DLKitAdapter()


_BUILDERS: Mapping[tuple[PreconditionerType, MatrixFormat], PreconditionerBuilder] = (
    MappingProxyType(
        {
            (PreconditionerType.IDENTITY, MatrixFormat.DENSE): _build_identity,
            (PreconditionerType.IDENTITY, MatrixFormat.CSR): _build_identity,
            (PreconditionerType.JACOBI, MatrixFormat.DENSE): _build_dense_jacobi,
            (PreconditionerType.JACOBI, MatrixFormat.CSR): _build_sparse_jacobi,
            (PreconditionerType.ILU, MatrixFormat.DENSE): _build_dense_ilu,
            (PreconditionerType.ILU, MatrixFormat.CSR): _build_sparse_ilu,
            (PreconditionerType.ICHOLESKY, MatrixFormat.DENSE): _build_dense_icholesky,
            (PreconditionerType.ICHOLESKY, MatrixFormat.CSR): _build_sparse_icholesky,
            (PreconditionerType.IC0, MatrixFormat.DENSE): _build_dense_ic0,
            (PreconditionerType.IC0, MatrixFormat.CSR): _build_sparse_ic0,
            (PreconditionerType.AMG, MatrixFormat.CSR): _build_sparse_amg,
            (PreconditionerType.ADAPTIVE_SA_AMG, MatrixFormat.DENSE): _build_dense_adaptive_sa,
            (PreconditionerType.ADAPTIVE_SA_AMG, MatrixFormat.CSR): _build_sparse_adaptive_sa,
            (PreconditionerType.BOOTSTRAP_AMG, MatrixFormat.DENSE): _build_dense_bootstrap_amg,
            (PreconditionerType.BOOTSTRAP_AMG, MatrixFormat.CSR): _build_sparse_bootstrap_amg,
            (PreconditionerType.NEURAL, MatrixFormat.DENSE): _build_neural,
            (PreconditionerType.NEURAL_AMG, MatrixFormat.DENSE): _build_neural_amg,
        }
    )
)
"""Single dispatch table from (preconditioner type, matrix format) to its builder.

Callable and LinearOperator preconditioners have no `PreconditionerType`
member today, so they are not reachable through this table.
"""

_DENSIFY_FOR_CSR: frozenset[PreconditionerType] = frozenset(
    {PreconditionerType.NEURAL, PreconditionerType.NEURAL_AMG}
)
"""Types whose only builder takes dense input; CSR requests densify explicitly first."""


def _resolve_format_input(
    matrix: torch.Tensor,
    preconditioner_type: PreconditionerType,
    matrix_format: MatrixFormat,
) -> tuple[torch.Tensor, MatrixFormat]:
    """Densify a CSR matrix for dense-only types, logging the O(n^2) cost once per call."""
    if matrix_format is MatrixFormat.CSR and preconditioner_type in _DENSIFY_FOR_CSR:
        n = matrix.shape[0]
        logger.info(
            f"Densifying CSR system matrix for {preconditioner_type.value} preconditioner "
            f"({n}x{n} dense; O(n^2) memory and time). Neural models take dense input only."
        )
        return matrix.to_dense(), MatrixFormat.DENSE
    return matrix, matrix_format


def _lookup_builder(
    preconditioner_type: PreconditionerType, matrix_format: MatrixFormat
) -> PreconditionerBuilder:
    """Return the builder for a (type, format) pair, or raise without falling back."""
    builder = _BUILDERS.get((preconditioner_type, matrix_format))
    if builder is None:
        raise ValueError(
            f"Unsupported preconditioner type {preconditioner_type!r} "
            f"for matrix format {matrix_format.value!r}: no builder is registered"
        )
    return builder


def create_preconditioner(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> Preconditioner:
    """Create a preconditioner from configuration for a matrix in the given format.

    Args:
        matrix: System matrix A: a dense tensor for `MatrixFormat.DENSE`, a
            sparse CSR tensor for `MatrixFormat.CSR`.
        config: Preconditioner configuration from TOML
        adapter: Optional adapter for neural preconditioner (DI for testing)
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing); defaults
            to `create_inference_predictor` from `platform.dlkit.inference_adapter`.
        matrix_format: Storage format of `matrix`. Defaults to dense.

    Returns:
        Preconditioner instance

    Raises:
        ValueError: If no builder is registered for the (type, format) pair.
    """
    preconditioner, _ = create_preconditioner_with_coarsening(
        matrix, config, adapter, inference_predictor_factory, matrix_format
    )
    return preconditioner


def create_preconditioner_with_coarsening(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: Callable[[Path, None], InferencePredictorPort] | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Create a preconditioner, also returning its coarsening strategy when it has one.

    This is the single construction path; `create_preconditioner` returns its first
    element. AMG's realized coarse dimension is only knowable from the coarsening
    strategy actually used to build the hierarchy, not from config alone (POD's
    ``rank`` is often an energy threshold; AMG's ``theta`` yields an emergent
    aggregate count). Diagnostics that need the realized dimension must reuse this
    coarsening object rather than fitting a second one.

    The sparse AMG preset builds its aggregation internally and does not expose a
    coarsening object, so for CSR AMG the coarsening is ``None``.

    Args:
        matrix: System matrix A, in the format named by `matrix_format`.
        config: Preconditioner configuration from TOML.
        adapter: Optional adapter for neural preconditioner (DI for testing).
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing).
        matrix_format: Storage format of `matrix`. Defaults to dense.

    Returns:
        The preconditioner, and its coarsening strategy if the dense AMG path
        built one (`None` for every other type and for sparse AMG).

    Raises:
        ValueError: If no builder is registered for the (type, format) pair.
        TypeError: If the config variant does not match its preconditioner type.
    """
    deps = _BuildDeps(adapter=adapter, inference_predictor_factory=inference_predictor_factory)
    operator, resolved_format = _resolve_format_input(matrix, config.type, matrix_format)
    if config.type is PreconditionerType.AMG and resolved_format is MatrixFormat.DENSE:
        if not isinstance(config, AMGPreconditionerConfig):
            raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
        build = _build_amg(operator, config, deps.inference_predictor_factory)
        return build.preconditioner, build.coarsening
    builder = _lookup_builder(config.type, resolved_format)
    return builder(operator, config, deps), None


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
