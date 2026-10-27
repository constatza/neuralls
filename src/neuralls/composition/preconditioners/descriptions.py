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
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import ScheduledPreconditioner
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    AggregationCoarsening,
    AMGPreconditioner,
    BootstrapAMGPreconditioner,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

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
from neuralls.shared.scientific_description import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BOOTSTRAP_CYCLES,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COARSE_DIMENSION,
    DAMPING,
    DROP_TOLERANCE,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SMOOTHER_DAMPING,
    TARGET_COARSE_DIMENSION,
    THRESHOLD,
    DecimalFact,
    DisplayTextFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
    ValidatedLabel,
)
from neuralls.shared.scientific_description.catalog import (
    BETA,
    CALIBER,
    CALIBER_GROWTH_PENALTY,
    COMPATIBLE_RELAXATION_SWEEPS,
    COMPATIBLE_RELAXATION_TOLERANCE,
    FALLBACK,
    ITERATION_LIMIT,
    MAX_COARSE_DIMENSION,
    METRIC,
    RANK,
    REALIZED_LEVELS,
    SEED,
    SNAPSHOT_COUNT,
    START_ITERATION,
    STEPS,
    USE_LSR,
    WEIGHTING,
    FactRole,
)

__all__ = [
    "FIELD_ROLES",
    "MissingArchitectureNameError",
    "describe_configured_preconditioner",
    "describe_realized_preconditioner",
]

type Facts = tuple[ScientificFact, ...]

_ESSENTIAL = FieldImportance.ESSENTIAL
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


def _int(key: FieldKey, value: int) -> IntegerFact:
    return IntegerFact(key, value, _ESSENTIAL)


def _dec(key: FieldKey, value: float) -> DecimalFact:
    return DecimalFact(key, value, _ESSENTIAL)


def _damping(key: FieldKey, value: float | None) -> ScientificFact:
    return TextFact(key, "auto", _ESSENTIAL) if value is None else _dec(key, value)


def _default_of(cfg: BaseModel, field_name: str) -> object:
    return type(cfg).model_fields[field_name].default


def _scheduling_facts(cfg: BasePreconditionerConfig) -> Facts:
    """Scheduling controls change the executed method, so only non-default ones appear."""
    facts: list[ScientificFact] = []
    if cfg.start_iter != _default_of(cfg, "start_iter"):
        facts.append(_int(START_ITERATION, cfg.start_iter))
    if cfg.limit_iters != _default_of(cfg, "limit_iters"):
        facts.append(_int(ITERATION_LIMIT, cfg.limit_iters))
    if cfg.fallback != _default_of(cfg, "fallback"):
        facts.append(TextFact(FALLBACK, cfg.fallback.value, _ESSENTIAL))
    return tuple(facts)


def _smoothing_facts(cfg: AMGPreconditionerConfig | NeuralAMGPreconditionerConfig) -> Facts:
    return (
        _int(PRE_SMOOTHING_STEPS, cfg.pre_smoothing_steps),
        _int(POST_SMOOTHING_STEPS, cfg.post_smoothing_steps),
        _damping(SMOOTHER_DAMPING, cfg.smoother_omega),
    )


def _architecture_fact(cfg: NeuralPreconditionerConfig) -> ScientificFact:
    if cfg.architecture is None:
        raise MissingArchitectureNameError
    return DisplayTextFact(MODEL_ARCHITECTURE, ValidatedLabel(cfg.architecture), _ESSENTIAL)


def _weighting_facts(
    weighting: RawWeightingConfig | PowerNormWeightingConfig | SmootherPersistenceWeightingConfig,
) -> Facts:
    match weighting:
        case RawWeightingConfig():
            return ()
        case PowerNormWeightingConfig():
            return (
                TextFact(WEIGHTING, weighting.method, _ESSENTIAL),
                TextFact(METRIC, weighting.metric, _ESSENTIAL),
                _dec(BETA, weighting.beta),
            )
        case SmootherPersistenceWeightingConfig():
            return (
                TextFact(WEIGHTING, weighting.method, _ESSENTIAL),
                _int(STEPS, weighting.steps),
                *(() if weighting.omega is None else (_dec(DAMPING, weighting.omega),)),
            )
        case _:
            assert_never(weighting)


def _pod_facts(cfg: PODCoarseningConfig | NeuralPODCoarseningConfig) -> Facts:
    rank = _int(RANK, cfg.rank) if isinstance(cfg.rank, int) else _dec(RANK, cfg.rank)
    snapshots = (
        () if cfg.n_snapshots == _ALL_SNAPSHOTS else (_int(SNAPSHOT_COUNT, cfg.n_snapshots),)
    )
    weighting = _weighting_facts(cfg.weighting) if isinstance(cfg, PODCoarseningConfig) else ()
    return (rank, *snapshots, *weighting)


def _amg_facts(cfg: AMGPreconditionerConfig) -> Facts:
    match cfg.coarsening:
        case PODCoarseningConfig() | NeuralPODCoarseningConfig() as pod:
            return (*_pod_facts(pod), *_smoothing_facts(cfg))
        case AggregationCoarseningConfig() as aggregation:
            return (
                _int(MULTIGRID_LEVELS, cfg.n_levels),
                _dec(THRESHOLD, aggregation.theta),
                *(() if aggregation.omega is None else (_dec(DAMPING, aggregation.omega),)),
                *_smoothing_facts(cfg),
            )
        case TargetDimCoarseningConfig() as target:
            return (
                _int(MULTIGRID_LEVELS, cfg.n_levels),
                _int(TARGET_COARSE_DIMENSION, target.target_coarse_dim),
                *(() if target.omega is None else (_dec(DAMPING, target.omega),)),
                *_smoothing_facts(cfg),
            )
        case _:
            assert_never(cfg.coarsening)


def _configured_facts(cfg: ConcretePreconditionerConfig) -> Facts:
    match cfg:
        case StandardPreconditionerConfig():
            return ()
        case IC0PreconditionerConfig():
            return (_dec(DROP_TOLERANCE, cfg.threshold),)
        case NeuralPreconditionerConfig():
            return (_architecture_fact(cfg),)
        case AMGPreconditionerConfig():
            return _amg_facts(cfg)
        case NeuralAMGPreconditionerConfig():
            return (
                _int(MULTIGRID_LEVELS, cfg.n_levels),
                *(() if cfg.aggregation_omega is None else (_dec(DAMPING, cfg.aggregation_omega),)),
                *_smoothing_facts(cfg),
            )
        case AdaptiveSAPreconditionerConfig():
            return (
                _int(MULTIGRID_LEVELS, cfg.n_levels),
                _int(CANDIDATE_COUNT, cfg.num_candidates),
                _int(CANDIDATE_ITERATIONS, cfg.candidate_iters),
                _int(MAX_COARSE_DIMENSION, cfg.max_coarse),
                _dec(THRESHOLD, cfg.theta),
                _dec(DAMPING, cfg.omega),
                _int(SEED, cfg.seed),
            )
        case BootstrapAMGPreconditionerConfig():
            return (
                _int(MULTIGRID_LEVELS, cfg.n_levels),
                _int(RELAXATION_TEST_VECTORS, cfg.k_r),
                _int(RELAXATION_SWEEPS, cfg.eta),
                _int(BOOTSTRAP_CYCLES, cfg.n_bootstrap_cycles),
                _int(COMPATIBLE_RELAXATION_SWEEPS, cfg.nu),
                _dec(COMPATIBLE_RELAXATION_TOLERANCE, cfg.delta),
                _dec(ALGEBRAIC_DISTANCE_THRESHOLD, cfg.theta_ad),
                _int(CALIBER, cfg.caliber),
                _dec(CALIBER_GROWTH_PENALTY, cfg.gamma),
                TextFact(USE_LSR, "on" if cfg.use_lsr else "off", _ESSENTIAL),
                _int(MAX_COARSE_DIMENSION, cfg.max_coarse),
                _int(SEED, cfg.seed),
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


@singledispatch
def _coarsening_facts(coarsening: CoarseningStrategy, matrix: torch.Tensor) -> Facts:
    return ()


@_coarsening_facts.register
def _(coarsening: PODCoarseningStrategy, matrix: torch.Tensor) -> Facts:
    return (_int(COARSE_DIMENSION, coarsening.rank),)


@_coarsening_facts.register
def _(coarsening: AggregationCoarsening, matrix: torch.Tensor) -> Facts:
    coarse_matrix, _ = coarsening.build_transfer(matrix)
    return (_int(COARSE_DIMENSION, coarse_matrix.shape[0]),)


@_coarsening_facts.register
def _(coarsening: TargetDimensionCoarsening, matrix: torch.Tensor) -> Facts:
    return (_int(COARSE_DIMENSION, coarsening.realized_coarse_dim(matrix)),)


@singledispatch
def describe_realized_preconditioner(precond: Preconditioner) -> Facts:
    """Facts only the built object knows; ``()`` for types with none."""
    return ()


@describe_realized_preconditioner.register
def _(precond: ScheduledPreconditioner) -> Facts:
    return describe_realized_preconditioner(precond.primary)


@describe_realized_preconditioner.register
def _(precond: AMGPreconditioner) -> Facts:
    return _coarsening_facts(precond.coarsening, precond.matrix)


def _hierarchy_facts(matrices: list[torch.Tensor]) -> Facts:
    return (
        _int(REALIZED_LEVELS, len(matrices)),
        _int(COARSE_DIMENSION, int(matrices[-1].shape[0])),
    )


@describe_realized_preconditioner.register
def _(precond: AdaptiveSAPreconditioner) -> Facts:
    return _hierarchy_facts(list(precond.result.matrices))


@describe_realized_preconditioner.register
def _(precond: BootstrapAMGPreconditioner) -> Facts:
    return _hierarchy_facts(list(precond.result.matrices))
