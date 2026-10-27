"""Scientific descriptions of preconditioner configs and built preconditioners.

Configured facts come from the validated config and are known before execution:
every algorithmic parameter (defaults included), reproducibility seeds, and
scheduling controls only when non-default. Realized facts come from the built
object after setup. Each emits facts in the order it wants them shown. Checkpoint,
assignment, path, and config-name identity is never a fact.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import singledispatch
from types import MappingProxyType
from typing import assert_never

import torch
from pydantic import BaseModel
from torchalg.multigrid.hierarchy import MultigridHierarchy
from torchalg.preconditioners.base import Preconditioner, PreconditionerNotReadyError
from torchalg.preconditioners.implementations import (
    IC0Preconditioner,
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    AggregationCoarsening,
    AMGPreconditioner,
    BootstrapAMGPreconditioner,
    NeuralCoarseningStrategy,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.coarsening import (
    AggregationCoarsening as SparseAggregationCoarsening,
)
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner as SparseIC0Preconditioner
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner as SparseILUPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)

from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BasePreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralCheckpointRef,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PowerNormWeightingConfig,
    RawWeightingConfig,
    SmootherPersistenceWeightingConfig,
    StandardPreconditionerConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner_family import preconditioner_family
from neuralls.shared.scientific_description.catalog import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BETA,
    BOOTSTRAP_CYCLES,
    CALIBER,
    CALIBER_GROWTH_PENALTY,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COARSE_DIMENSION,
    COMPATIBLE_RELAXATION_SWEEPS,
    COMPATIBLE_RELAXATION_TOLERANCE,
    DAMPING,
    DROP_TOLERANCE,
    FALLBACK,
    ITERATION_LIMIT,
    MAX_COARSE_DIMENSION,
    METRIC,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RANK,
    REALIZED_LEVELS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SEED,
    SMOOTHER_DAMPING,
    SNAPSHOT_COUNT,
    START_ITERATION,
    STEPS,
    TARGET_COARSE_DIMENSION,
    THRESHOLD,
    USE_LSR,
    WEIGHTING,
    FactRole,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    FieldKey,
    ScientificDescription,
    ScientificFact,
    ValidatedLabel,
)

__all__ = [
    "FIELD_ROLES",
    "MissingArchitectureNameError",
    "UnsupportedRealizedPreconditionerError",
    "describe_configured_preconditioner",
    "describe_realized_preconditioner",
]

type Facts = tuple[ScientificFact, ...]
type PODStrategy = PODCoarseningStrategy | SparsePODCoarseningStrategy

_ALL_SNAPSHOTS = -1
"""``n_snapshots`` value meaning every matched snapshot; not a count worth showing."""


class MissingArchitectureNameError(ValueError):
    """A neural preconditioner has no ``architecture`` name."""

    def __init__(self) -> None:
        super().__init__(
            "A neural preconditioner needs a short 'architecture' name (for example 'Graph Net') "
            "so its architecture can be described; set it on the neural preconditioner or on "
            "the training job's [[jobs]] entry. Class names and job titles are never used."
        )


_A = FactRole.ALGORITHMIC
_R = FactRole.REPRODUCIBILITY
_S = FactRole.SCHEDULING
_O = FactRole.OPERATIONAL
_I = FactRole.IDENTITY
_P = FactRole.PROVENANCE

_SCHEDULING_ROLES = {"start_iter": _S, "limit_iters": _S, "fallback": _S, "type": _I}
_CHECKPOINT_ROLES = dict.fromkeys(NeuralCheckpointRef.model_fields, _I)

FIELD_ROLES: Mapping[type[BaseModel], Mapping[str, FactRole]] = MappingProxyType(
    {
        StandardPreconditionerConfig: _SCHEDULING_ROLES,
        IC0PreconditionerConfig: {**_SCHEDULING_ROLES, "threshold": _A},
        NeuralPreconditionerConfig: {
            **_SCHEDULING_ROLES,
            **_CHECKPOINT_ROLES,
            "architecture": _A,
            "extra_input_names": _O,
        },
        AMGPreconditionerConfig: {
            **_SCHEDULING_ROLES,
            "n_levels": _A,
            "pre_smoothing_steps": _A,
            "post_smoothing_steps": _A,
            "smoother_omega": _A,
            "coarsening": _A,
        },
        NeuralAMGPreconditionerConfig: {
            **_SCHEDULING_ROLES,
            "n_levels": _A,
            "pre_smoothing_steps": _A,
            "post_smoothing_steps": _A,
            "smoother_omega": _A,
            "aggregation_omega": _A,
            "prolongation": _I,
            "restriction": _I,
        },
        AdaptiveSAPreconditionerConfig: {
            **_SCHEDULING_ROLES,
            "n_levels": _A,
            "num_candidates": _A,
            "candidate_iters": _A,
            "max_coarse": _A,
            "theta": _A,
            "omega": _A,
            "seed": _R,
        },
        BootstrapAMGPreconditionerConfig: {
            **_SCHEDULING_ROLES,
            "n_levels": _A,
            "k_r": _A,
            "eta": _A,
            "n_bootstrap_cycles": _A,
            "nu": _A,
            "delta": _A,
            "theta_ad": _A,
            "caliber": _A,
            "gamma": _A,
            "use_lsr": _A,
            "max_coarse": _A,
            "seed": _R,
        },
        AggregationCoarseningConfig: {"method": _I, "theta": _A, "omega": _A},
        TargetDimCoarseningConfig: {
            "method": _I,
            "target_coarse_dim": _A,
            "theta_min": _O,
            "theta_max": _O,
            "step": _O,
            "omega": _A,
        },
        PODCoarseningConfig: {
            **_CHECKPOINT_ROLES,
            "method": _I,
            "dataset_dir": _P,
            "n_snapshots": _A,
            "rank": _A,
            "weighting": _A,
        },
        NeuralPODCoarseningConfig: {
            **_CHECKPOINT_ROLES,
            "method": _I,
            "dataset_dir": _P,
            "input_names": _O,
            "n_snapshots": _A,
            "rank": _A,
        },
        RawWeightingConfig: {"method": _A},
        PowerNormWeightingConfig: {"method": _A, "metric": _A, "beta": _A},
        SmootherPersistenceWeightingConfig: {"method": _A, "omega": _A, "steps": _A},
    }
)
"""Every config field's role. A newly added field fails the exhaustiveness test until it is
classified here, so it cannot silently disappear from labels (algorithmic fields) or leak into
them (identity, provenance, operational fields)."""


def _damping(key: FieldKey, value: float | None) -> ScientificFact:
    return make_fact(key, "auto" if value is None else value)


def _default_of(cfg: BaseModel, field_name: str) -> object:
    return type(cfg).model_fields[field_name].default


def _scheduling_facts(cfg: BasePreconditionerConfig) -> Facts:
    """Scheduling controls change the executed method, so only non-default ones appear."""
    facts: list[ScientificFact] = []
    if cfg.start_iter != _default_of(cfg, "start_iter"):
        facts.append(make_fact(START_ITERATION, cfg.start_iter))
    if cfg.limit_iters != _default_of(cfg, "limit_iters"):
        facts.append(make_fact(ITERATION_LIMIT, cfg.limit_iters))
    if cfg.fallback != _default_of(cfg, "fallback"):
        facts.append(make_fact(FALLBACK, cfg.fallback.value))
    return tuple(facts)


def _smoothing_facts(cfg: AMGPreconditionerConfig | NeuralAMGPreconditionerConfig) -> Facts:
    return (
        make_fact(PRE_SMOOTHING_STEPS, cfg.pre_smoothing_steps),
        make_fact(POST_SMOOTHING_STEPS, cfg.post_smoothing_steps),
        _damping(SMOOTHER_DAMPING, cfg.smoother_omega),
    )


def _architecture_fact(cfg: NeuralPreconditionerConfig) -> ScientificFact:
    if cfg.architecture is None:
        raise MissingArchitectureNameError
    return make_fact(MODEL_ARCHITECTURE, ValidatedLabel(cfg.architecture))


def _weighting_facts(
    weighting: RawWeightingConfig | PowerNormWeightingConfig | SmootherPersistenceWeightingConfig,
) -> Facts:
    match weighting:
        case RawWeightingConfig():
            return ()
        case PowerNormWeightingConfig():
            return (
                make_fact(WEIGHTING, weighting.method),
                make_fact(METRIC, weighting.metric),
                make_fact(BETA, weighting.beta),
            )
        case SmootherPersistenceWeightingConfig():
            return (
                make_fact(WEIGHTING, weighting.method),
                make_fact(STEPS, weighting.steps),
                *(() if weighting.omega is None else (make_fact(DAMPING, weighting.omega),)),
            )
        case _:
            assert_never(weighting)


def _pod_facts(cfg: PODCoarseningConfig | NeuralPODCoarseningConfig) -> Facts:
    rank = make_fact(RANK, cfg.rank)
    snapshots = (
        () if cfg.n_snapshots == _ALL_SNAPSHOTS else (make_fact(SNAPSHOT_COUNT, cfg.n_snapshots),)
    )
    weighting = _weighting_facts(cfg.weighting) if isinstance(cfg, PODCoarseningConfig) else ()
    return (rank, *snapshots, *weighting)


def _amg_facts(cfg: AMGPreconditionerConfig) -> Facts:
    match cfg.coarsening:
        case PODCoarseningConfig() | NeuralPODCoarseningConfig() as pod:
            return (*_pod_facts(pod), *_smoothing_facts(cfg))
        case AggregationCoarseningConfig() as aggregation:
            return (
                make_fact(MULTIGRID_LEVELS, cfg.n_levels),
                make_fact(THRESHOLD, aggregation.theta),
                *(() if aggregation.omega is None else (make_fact(DAMPING, aggregation.omega),)),
                *_smoothing_facts(cfg),
            )
        case TargetDimCoarseningConfig() as target:
            return (
                make_fact(MULTIGRID_LEVELS, cfg.n_levels),
                make_fact(TARGET_COARSE_DIMENSION, target.target_coarse_dim),
                *(() if target.omega is None else (make_fact(DAMPING, target.omega),)),
                *_smoothing_facts(cfg),
            )
        case _:
            assert_never(cfg.coarsening)


def _configured_facts(cfg: ConcretePreconditionerConfig) -> Facts:
    match cfg:
        case StandardPreconditionerConfig():
            return ()
        case IC0PreconditionerConfig():
            return (make_fact(DROP_TOLERANCE, cfg.threshold),)
        case NeuralPreconditionerConfig():
            return (_architecture_fact(cfg),)
        case AMGPreconditionerConfig():
            return _amg_facts(cfg)
        case NeuralAMGPreconditionerConfig():
            return (
                make_fact(MULTIGRID_LEVELS, cfg.n_levels),
                *(
                    ()
                    if cfg.aggregation_omega is None
                    else (make_fact(DAMPING, cfg.aggregation_omega),)
                ),
                *_smoothing_facts(cfg),
            )
        case AdaptiveSAPreconditionerConfig():
            return (
                make_fact(MULTIGRID_LEVELS, cfg.n_levels),
                make_fact(CANDIDATE_COUNT, cfg.num_candidates),
                make_fact(CANDIDATE_ITERATIONS, cfg.candidate_iters),
                make_fact(MAX_COARSE_DIMENSION, cfg.max_coarse),
                make_fact(THRESHOLD, cfg.theta),
                make_fact(DAMPING, cfg.omega),
                make_fact(SEED, cfg.seed),
            )
        case BootstrapAMGPreconditionerConfig():
            return (
                make_fact(MULTIGRID_LEVELS, cfg.n_levels),
                make_fact(RELAXATION_TEST_VECTORS, cfg.k_r),
                make_fact(RELAXATION_SWEEPS, cfg.eta),
                make_fact(BOOTSTRAP_CYCLES, cfg.n_bootstrap_cycles),
                make_fact(COMPATIBLE_RELAXATION_SWEEPS, cfg.nu),
                make_fact(COMPATIBLE_RELAXATION_TOLERANCE, cfg.delta),
                make_fact(ALGEBRAIC_DISTANCE_THRESHOLD, cfg.theta_ad),
                make_fact(CALIBER, cfg.caliber),
                make_fact(CALIBER_GROWTH_PENALTY, cfg.gamma),
                make_fact(USE_LSR, "on" if cfg.use_lsr else "off"),
                make_fact(MAX_COARSE_DIMENSION, cfg.max_coarse),
                make_fact(SEED, cfg.seed),
            )
        case _:
            assert_never(cfg)


def describe_configured_preconditioner(cfg: ConcretePreconditionerConfig) -> ScientificDescription:
    """Describe ``cfg`` completely from configuration alone, with no I/O or setup."""
    return ScientificDescription(
        base_name=preconditioner_family(cfg).abbreviation(),
        facts=(*_configured_facts(cfg), *_scheduling_facts(cfg)),
    )


# ==============================================================================
# Realized facts — read from the constructed object
# ==============================================================================


class UnsupportedRealizedPreconditionerError(TypeError):
    """A built preconditioner or coarsening class has no declared realized facts.

    Every class this repository builds declares its realized facts, possibly none. A class
    that does not is a new, undescribed method, so describing it fails instead of
    silently labelling it without its realized facts.
    """

    def __init__(self, kind: type) -> None:
        self.kind = kind
        super().__init__(
            f"No realized facts are declared for {kind.__module__}.{kind.__qualname__}; "
            "register its realized facts (an empty tuple when it has none)"
        )


def _first_coarse_dimension(precond: AMGPreconditioner) -> int:
    """Dimension of the first coarse level, read from the hierarchy ``setup`` already built.

    torchalg exposes no public accessor for the built hierarchy of a plain
    ``AMGPreconditioner``; reading it avoids building a transfer a second time, which would
    repeat the setup cost per matrix outside the timed and failure-handled setup.
    """
    match precond._hierarchy:
        case MultigridHierarchy(levels=(_, coarse, *_)):
            return int(coarse.matrix.shape[0])
        case _:
            raise PreconditionerNotReadyError(
                "A multigrid hierarchy with a coarse level is required to describe the "
                "realized coarse dimension; call setup() first"
            )


@singledispatch
def _coarsening_facts(coarsening: CoarseningStrategy, precond: AMGPreconditioner) -> Facts:
    raise UnsupportedRealizedPreconditionerError(type(coarsening))


@_coarsening_facts.register(PODCoarseningStrategy)
@_coarsening_facts.register(SparsePODCoarseningStrategy)
def _(coarsening: PODStrategy, precond: AMGPreconditioner) -> Facts:
    return (make_fact(COARSE_DIMENSION, coarsening.rank),)


@_coarsening_facts.register(AggregationCoarsening)
@_coarsening_facts.register(SparseAggregationCoarsening)
def _(
    coarsening: AggregationCoarsening | SparseAggregationCoarsening, precond: AMGPreconditioner
) -> Facts:
    return (make_fact(COARSE_DIMENSION, _first_coarse_dimension(precond)),)


@_coarsening_facts.register
def _(coarsening: TargetDimensionCoarsening, precond: AMGPreconditioner) -> Facts:
    return (make_fact(COARSE_DIMENSION, coarsening.realized_coarse_dim(precond.matrix)),)


@_coarsening_facts.register
def _(coarsening: NeuralCoarseningStrategy, precond: AMGPreconditioner) -> Facts:
    return ()


@singledispatch
def describe_realized_preconditioner(precond: Preconditioner) -> Facts:
    """Facts only the built object knows.

    Raises:
        UnsupportedRealizedPreconditionerError: If the class declares no realized facts.
    """
    raise UnsupportedRealizedPreconditionerError(type(precond))


@describe_realized_preconditioner.register(Identity)
@describe_realized_preconditioner.register(JacobiPreconditioner)
@describe_realized_preconditioner.register(SparseJacobiPreconditioner)
@describe_realized_preconditioner.register(ILUPreconditioner)
@describe_realized_preconditioner.register(SparseILUPreconditioner)
@describe_realized_preconditioner.register(IC0Preconditioner)
@describe_realized_preconditioner.register(SparseIC0Preconditioner)
@describe_realized_preconditioner.register(NeuralPreconditioner)
def _(precond: Preconditioner) -> Facts:
    return ()


@describe_realized_preconditioner.register
def _(precond: ScheduledPreconditioner) -> Facts:
    return describe_realized_preconditioner(precond.primary)


@describe_realized_preconditioner.register
def _(precond: AMGPreconditioner) -> Facts:
    return _coarsening_facts(precond.coarsening, precond)


def _hierarchy_facts(matrices: list[torch.Tensor]) -> Facts:
    return (
        make_fact(REALIZED_LEVELS, len(matrices)),
        make_fact(COARSE_DIMENSION, int(matrices[-1].shape[0])),
    )


@describe_realized_preconditioner.register(AdaptiveSAPreconditioner)
@describe_realized_preconditioner.register(SparseAdaptiveSAPreconditioner)
@describe_realized_preconditioner.register(BootstrapAMGPreconditioner)
@describe_realized_preconditioner.register(SparseBootstrapAMGPreconditioner)
def _(
    precond: AdaptiveSAPreconditioner
    | SparseAdaptiveSAPreconditioner
    | BootstrapAMGPreconditioner
    | SparseBootstrapAMGPreconditioner,
) -> Facts:
    return _hierarchy_facts(list(precond.result.matrices))
