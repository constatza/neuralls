"""Per-(type, format) preconditioner builders, dispatched through one table.

Every builder returns `(Preconditioner, CoarseningStrategy | None)`: the table is
the single dispatch mechanism, with no special case for any one entry. Only AMG
exposes a real coarsening strategy (needed by diagnostics to read back its
realized coarse dimension); every other builder's second element is `None`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
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
    AdaptiveSAPreconditioner,
    AMGPreconditioner,
    BootstrapAMGPreconditioner,
    JacobiSmoother,
    NeuralCoarseningStrategy,
    VCycle,
)
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

from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter
from neuralls.shared.types import MatrixFormat

from .coarsening import PredictorFactory, _build_amg_coarsening

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
    from torchalg.preconditioners.ports import PredictorAdapter

    from neuralls.platform.config.models.preconditioner import ConcretePreconditionerConfig


@dataclass(frozen=True)
class _BuildDeps:
    """Injected collaborators shared by every preconditioner builder."""

    adapter: PredictorAdapter | None
    inference_predictor_factory: PredictorFactory | None


type PreconditionerBuilder = Callable[
    [torch.Tensor, ConcretePreconditionerConfig, _BuildDeps],
    tuple[Preconditioner, CoarseningStrategy | None],
]
"""Builds one preconditioner from a matrix already in the builder's format."""


def _build_identity(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Return the no-op preconditioner; it ignores the matrix in either format."""
    del matrix, config, deps
    return Identity(), None


def _build_dense_jacobi(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense diagonal (Jacobi) preconditioner."""
    del config, deps
    return JacobiPreconditioner().setup(matrix), None


def _build_sparse_jacobi(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR diagonal (Jacobi) preconditioner."""
    del config, deps
    return SparseJacobiPreconditioner().setup(matrix), None


def _build_dense_ilu(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense ILU(0) preconditioner."""
    del config, deps
    return ILUPreconditioner().setup(matrix), None


def _build_sparse_ilu(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR ILU(0) preconditioner."""
    del config, deps
    return SparseILUPreconditioner().setup(matrix), None


def _build_dense_icholesky(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense incomplete-Cholesky preconditioner over a supplied factor."""
    del config, deps
    return ICholeskyPreconditioner().setup(matrix), None


def _build_sparse_icholesky(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR incomplete-Cholesky preconditioner over a supplied factor."""
    del config, deps
    return SparseICholeskyPreconditioner().setup(matrix), None


def _build_dense_ic0(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense IC(0) preconditioner with its drop threshold."""
    del deps
    if not isinstance(config, IC0PreconditionerConfig):
        raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
    return IC0Preconditioner(threshold=config.threshold).setup(matrix), None


def _build_sparse_ic0(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR IC(0) preconditioner with its drop threshold."""
    del deps
    if not isinstance(config, IC0PreconditionerConfig):
        raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
    return SparseIC0Preconditioner(threshold=config.threshold).setup(matrix), None


def _build_sparse_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR AMG: A stays CSR in every level and in the transfer operators.

    Aggregation coarsening uses torchalg's sparse V-cycle preset. POD-2G and neural
    POD-2G use the same multigrid engine with torchalg's sparse POD strategy, so the
    coarse operator is formed as P.T A P without densifying A. Target-dimension
    coarsening has no sparse counterpart and is rejected. The sparse AMG preset builds
    its aggregation internally and does not expose a coarsening object, so this always
    returns `None` for the coarsening, even on the POD/neural-POD path.
    """
    if not isinstance(config, AMGPreconditionerConfig):
        raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
    if isinstance(config.coarsening, AggregationCoarseningConfig):
        return (
            sparse_vcycle_amg(
                matrix,
                theta=config.coarsening.theta,
                omega=config.coarsening.omega,
                n_levels=config.n_levels,
                smoother_omega=config.smoother_omega,
                n_pre=config.pre_smoothing_steps,
                n_post=config.post_smoothing_steps,
            ).setup(matrix),
            None,
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
    preconditioner = MultigridAMGPreconditioner(
        matrix=matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=True
    ).setup(matrix)
    return preconditioner, None


def _build_amg(
    matrix: torch.Tensor,
    config: AMGPreconditionerConfig,
    inference_predictor_factory: PredictorFactory | None = None,
) -> tuple[Preconditioner, CoarseningStrategy]:
    """Assemble an `AMGPreconditioner` and return it alongside its coarsening strategy.

    `AMGPreconditioner` keeps its `coarsening` as a private attribute (it is
    runtime state, not part of its public API), so diagnostics code that
    needs the coarsening strategy itself (e.g. to read back the realized
    coarse dimension via its public `build_transfer`) cannot get it from the
    preconditioner after the fact without reaching into private state.
    Returning both from the same build call means diagnostics reuse the
    exact coarsening object already used to build the hierarchy — no
    duplicate POD fit, no duplicate checkpoint load.

    Args:
        matrix: System matrix A.
        config: AMG preconditioner configuration.
        inference_predictor_factory: Optional batch-inference predictor factory
            for neural POD-2G coarsening (DI for testing).

    Returns:
        The assembled preconditioner and the coarsening strategy used to build it.
    """
    coarsening = _build_amg_coarsening(matrix, config, inference_predictor_factory)
    smoother = JacobiSmoother(omega=config.smoother_omega)
    cycle = VCycle(
        smoother=smoother,
        n_pre=config.pre_smoothing_steps,
        n_post=config.post_smoothing_steps,
    )
    preconditioner = AMGPreconditioner(
        matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=True
    ).setup(matrix)
    return preconditioner, coarsening


def _build_dense_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense AMG: the one builder whose coarsening is exposed in its return."""
    if not isinstance(config, AMGPreconditionerConfig):
        raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
    return _build_amg(matrix, config, deps.inference_predictor_factory)


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
    return AdaptiveSAPreconditioner(
        num_candidates=config.num_candidates,
        candidate_iters=config.candidate_iters,
        max_levels=config.n_levels,
        max_coarse=config.max_coarse,
        theta=config.theta,
        omega=config.omega,
        seed=config.seed,
    ).setup(matrix)


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
    return BootstrapAMGPreconditioner(
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
    ).setup(matrix)


def _build_dense_adaptive_sa(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense alpha-SA AMG preconditioner."""
    del deps
    if not isinstance(config, AdaptiveSAPreconditionerConfig):
        raise TypeError(
            f"ADAPTIVE_SA_AMG type requires AdaptiveSAPreconditionerConfig, got {type(config)}"
        )
    return _build_adaptive_sa(matrix, config), None


def _build_sparse_adaptive_sa(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR alpha-SA AMG preconditioner."""
    del deps
    if not isinstance(config, AdaptiveSAPreconditionerConfig):
        raise TypeError(
            f"ADAPTIVE_SA_AMG type requires AdaptiveSAPreconditionerConfig, got {type(config)}"
        )
    return (
        SparseAdaptiveSAPreconditioner(
            num_candidates=config.num_candidates,
            candidate_iters=config.candidate_iters,
            max_levels=config.n_levels,
            max_coarse=config.max_coarse,
            theta=config.theta,
            omega=config.omega,
            seed=config.seed,
        ).setup(matrix),
        None,
    )


def _build_dense_bootstrap_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Dense bootstrap AMG (BAMG) preconditioner."""
    del deps
    if not isinstance(config, BootstrapAMGPreconditionerConfig):
        raise TypeError(
            f"BOOTSTRAP_AMG type requires BootstrapAMGPreconditionerConfig, got {type(config)}"
        )
    return _build_bootstrap_amg(matrix, config), None


def _build_sparse_bootstrap_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Sparse CSR bootstrap AMG (BAMG) preconditioner."""
    del deps
    if not isinstance(config, BootstrapAMGPreconditionerConfig):
        raise TypeError(
            f"BOOTSTRAP_AMG type requires BootstrapAMGPreconditionerConfig, got {type(config)}"
        )
    return (
        SparseBootstrapAMGPreconditioner(
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
        ).setup(matrix),
        None,
    )


def _build_neural(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Checkpoint-backed neural preconditioner; `setup()` ignores the matrix, only loads the model."""
    if not isinstance(config, NeuralPreconditionerConfig):
        raise TypeError(f"Neural type requires NeuralPreconditionerConfig, got {type(config)}")
    ckpt = config.active_checkpoint_path
    if ckpt is None:
        raise ValueError(
            "NeuralPreconditionerConfig requires checkpoint_path or resolved_checkpoint_path"
        )
    return (
        NeuralPreconditioner(
            checkpoint_path=ckpt,
            config_path=config.config_path,
            data_config_path=config.data_config_path,
            adapter=_resolve_adapter(deps),
            extra_input_names=tuple(config.extra_input_names),
        ).setup(matrix),
        None,
    )


def _build_neural_amg(
    matrix: torch.Tensor, config: ConcretePreconditionerConfig, deps: _BuildDeps
) -> tuple[Preconditioner, CoarseningStrategy | None]:
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
    preconditioner = AMGPreconditioner(
        matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=False
    ).setup(matrix)
    return preconditioner, None


def _resolve_adapter(deps: _BuildDeps) -> PredictorAdapter:
    """Return the injected predictor adapter, or the DLKit default when none was injected."""
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
            (PreconditionerType.AMG, MatrixFormat.DENSE): _build_dense_amg,
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
member today, so they are not reachable through this table. Dense AMG is a
table entry like any other — no special case intercepts it beforehand.
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
