"""Display names for preconditioners: a family abbreviation plus scientific facts.

A label is ``render(ScientificDescription(base, facts), LEGEND_PROFILE, policy)``
— the same vocabulary, notation policy (plain / mathtext / latex, symbol
overrides) and length budget the rest of the study's descriptions use. Facts
come from two places, kept apart on purpose:

* **Realized facts** are read from the *constructed* object, because it is the
  ground truth for what was actually built (the POD basis's real fitted rank,
  the coarse dimension an aggregation threshold actually produced, an
  adaptive hierarchy's realized level count, the neural model's architecture).
  They are ``ESSENTIAL``: never dropped to fit the budget. Dispatch is
  ``functools.singledispatch`` on the object's class, so the most-derived
  registration wins by MRO and a new preconditioner registers its own describer.
* **Configured facts** come from the config (``configured_facts``): the grid
  level count, the IC0 drop tolerance, smoothing steps, and the aSA/BAMG
  settings. torchalg keeps cycles and smoothers private, so the config is the
  only reliable source, and it covers CSR (sparse) preconditioners too. Anything
  equal to its config default is omitted, so a default configuration keeps a
  short legend; the settings that merely distinguish variants are ``VARIANT``
  and are the first facts dropped when the legend budget is exceeded. Seeds are
  never shown.

The base display name (``"POD-2G"``, ``"aSA"``, ``"BAMG"``) comes from the
preconditioner's ``PreconditionerFamilyKey``, not from the config's ``name``:
``name`` is a derived ASCII identity slug used as the dict/cache key and MLflow
metric-key segment, so it must stay ASCII and run-stable, which this
human-readable render is free to ignore. Workflow provenance stays a separate
input appended with ``contextualize_preconditioner_label``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import singledispatch
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
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.models.preconditioner_family import (
    PreconditionerFamilyKey,
    preconditioner_family,
)
from neuralls.platform.dlkit.predictor_adapter import DLKitNeuralPreconditioner
from neuralls.shared.scientific_description import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BOOTSTRAP_CYCLES,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COARSE_DIMENSION,
    COARSENING_STRATEGY,
    DAMPING,
    DROP_TOLERANCE,
    LEGEND_PROFILE,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SMOOTHER_DAMPING,
    THRESHOLD,
    DecimalFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    NotationPolicy,
    ScientificDescription,
    ScientificFact,
    TextFact,
    render,
    render_facts,
)

__all__ = [
    "build_configured_preconditioner_label",
    "build_preconditioner_labels",
    "coarsening_facts",
    "configured_facts",
    "contextualize_preconditioner_label",
    "describe_preconditioner",
    "preconditioner_label",
    "realized_facts",
]

type Facts = tuple[ScientificFact, ...]

_ESSENTIAL = FieldImportance.ESSENTIAL
_VARIANT = FieldImportance.VARIANT

_PRECONDITIONER_LEGEND = replace(
    LEGEND_PROFILE,
    precision={
        THRESHOLD: 2,
        DAMPING: 2,
        SMOOTHER_DAMPING: 2,
        ALGEBRAIC_DISTANCE_THRESHOLD: 2,
        DROP_TOLERANCE: 1,
    },
)
"""The shared legend profile (55-character budget, VARIANT facts dropped first) with
the precision the study's plots already use for these quantities."""


def _default_policy(policy: NotationPolicy | None) -> NotationPolicy:
    return policy if policy is not None else NotationPolicy.default()


# ==============================================================================
# Realized facts — read from the constructed object
# ==============================================================================


@singledispatch
def coarsening_facts(coarsening: CoarseningStrategy, matrix: torch.Tensor) -> Facts:
    """Realized facts of a coarsening strategy; an unknown strategy names its class.

    The set of torchalg strategies is open, so an unrecognized one is shown
    explicitly as ``coarsening=<ClassName>`` rather than silently dropped.

    Args:
        coarsening (CoarseningStrategy): The AMG preconditioner's strategy.
        matrix (torch.Tensor): Finest-level matrix, used to read back realized
            dimensions that a threshold only implies.
    """
    return (TextFact(COARSENING_STRATEGY, type(coarsening).__name__, _VARIANT),)


@coarsening_facts.register
def _(coarsening: PODCoarseningStrategy, matrix: torch.Tensor) -> Facts:
    return (IntegerFact(COARSE_DIMENSION, coarsening.rank, _ESSENTIAL),)


@coarsening_facts.register
def _(coarsening: TargetDimensionCoarsening, matrix: torch.Tensor) -> Facts:
    return (IntegerFact(COARSE_DIMENSION, coarsening.realized_coarse_dim(matrix), _ESSENTIAL),)


@coarsening_facts.register
def _(coarsening: AggregationCoarsening, matrix: torch.Tensor) -> Facts:
    coarse_matrix, _ = coarsening.build_transfer(matrix)
    omega = coarsening.omega
    omega_value = omega.item() if isinstance(omega, torch.Tensor) else omega
    damping: ScientificFact = (
        TextFact(DAMPING, "auto", _ESSENTIAL)
        if omega_value is None
        else DecimalFact(DAMPING, omega_value, _ESSENTIAL)
    )
    return (
        DecimalFact(THRESHOLD, coarsening.theta, _ESSENTIAL),
        damping,
        IntegerFact(COARSE_DIMENSION, coarse_matrix.shape[0], _ESSENTIAL),
    )


@singledispatch
def realized_facts(precond: Preconditioner) -> Facts:
    """Facts only the built object knows; ``()`` for types with none (Identity, Jacobi, ILU, IC0)."""
    return ()


@realized_facts.register
def _(precond: ScheduledPreconditioner) -> Facts:
    return realized_facts(precond.primary)


@realized_facts.register
def _(precond: AMGPreconditioner) -> Facts:
    return coarsening_facts(precond.coarsening, precond.matrix)


def _hierarchy_facts(n_levels: int, candidate_fact: IntegerFact, coarse_dimension: int) -> Facts:
    """Realized hierarchy of a bootstrapped/adaptive AMG: levels, its candidate count, coarsest size."""
    return (
        IntegerFact(MULTIGRID_LEVELS, n_levels, _ESSENTIAL),
        candidate_fact,
        IntegerFact(COARSE_DIMENSION, coarse_dimension, _ESSENTIAL),
    )


@realized_facts.register
def _(precond: AdaptiveSAPreconditioner) -> Facts:
    result = precond.result
    return _hierarchy_facts(
        len(result.matrices),
        IntegerFact(CANDIDATE_COUNT, result.candidates.shape[1], _ESSENTIAL),
        int(result.matrices[-1].shape[0]),
    )


@realized_facts.register
def _(precond: BootstrapAMGPreconditioner) -> Facts:
    result = precond.result
    return _hierarchy_facts(
        len(result.matrices),
        IntegerFact(RELAXATION_TEST_VECTORS, result.candidates.shape[1], _ESSENTIAL),
        int(result.matrices[-1].shape[0]),
    )


@realized_facts.register
def _(precond: DLKitNeuralPreconditioner) -> Facts:
    return (TextFact(MODEL_ARCHITECTURE, precond.model_class_name, _ESSENTIAL),)


# ==============================================================================
# Configured facts — read from the config, defaults omitted
# ==============================================================================


def _overridden(cfg: BaseModel, field_name: str) -> bool:
    """Whether `field_name` differs from its declared config default."""
    return getattr(cfg, field_name) != type(cfg).model_fields[field_name].default


def _integer_variant(cfg: BaseModel, field_name: str, key: FieldKey) -> Facts:
    if not _overridden(cfg, field_name):
        return ()
    return (IntegerFact(key, getattr(cfg, field_name), _VARIANT),)


def _decimal_variant(cfg: BaseModel, field_name: str, key: FieldKey) -> Facts:
    if not _overridden(cfg, field_name):
        return ()
    return (DecimalFact(key, getattr(cfg, field_name), _VARIANT),)


def _smoothing_facts(cfg: AMGPreconditionerConfig | NeuralAMGPreconditionerConfig) -> Facts:
    return (
        *_integer_variant(cfg, "pre_smoothing_steps", PRE_SMOOTHING_STEPS),
        *_integer_variant(cfg, "post_smoothing_steps", POST_SMOOTHING_STEPS),
        *_decimal_variant(cfg, "smoother_omega", SMOOTHER_DAMPING),
    )


def configured_facts(cfg: ConcretePreconditionerConfig) -> Facts:
    """Facts that come from the configuration, with default-valued ones omitted.

    Exhaustive over the closed config union, so a new preconditioner type forces
    a decision here. The grid level count is ``ESSENTIAL`` for aggregation-style
    AMG (POD-2G is architecturally single-basis two-grid, so a level count there
    would be constant noise); everything that only tells variants apart is
    ``VARIANT``. Seeds are never facts.

    Args:
        cfg (ConcretePreconditionerConfig): The preconditioner's config.
    """
    match cfg:
        case StandardPreconditionerConfig() | NeuralPreconditionerConfig():
            return ()
        case IC0PreconditionerConfig():
            return (
                (DecimalFact(DROP_TOLERANCE, cfg.threshold, _ESSENTIAL),)
                if _overridden(cfg, "threshold")
                else ()
            )
        case AMGPreconditionerConfig():
            levels: Facts = (
                ()
                if isinstance(cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig)
                else (IntegerFact(MULTIGRID_LEVELS, cfg.n_levels, _ESSENTIAL),)
            )
            return (*levels, *_smoothing_facts(cfg))
        case NeuralAMGPreconditionerConfig():
            return (IntegerFact(MULTIGRID_LEVELS, cfg.n_levels, _ESSENTIAL), *_smoothing_facts(cfg))
        case AdaptiveSAPreconditionerConfig():
            return (
                *_decimal_variant(cfg, "theta", THRESHOLD),
                *_decimal_variant(cfg, "omega", DAMPING),
                *_integer_variant(cfg, "candidate_iters", CANDIDATE_ITERATIONS),
            )
        case BootstrapAMGPreconditionerConfig():
            return (
                *_integer_variant(cfg, "eta", RELAXATION_SWEEPS),
                *_integer_variant(cfg, "n_bootstrap_cycles", BOOTSTRAP_CYCLES),
                *_decimal_variant(cfg, "theta_ad", ALGEBRAIC_DISTANCE_THRESHOLD),
            )
        case _:
            assert_never(cfg)


# ==============================================================================
# Labels
# ==============================================================================


def describe_preconditioner(
    precond: Preconditioner, *, policy: NotationPolicy | None = None
) -> str:
    """Render only a built object's realized facts, e.g. ``"θ=0.25, ω=0.67, c=2"``.

    Args:
        precond (Preconditioner): A constructed preconditioner; schedule
            wrappers are unwrapped to their primary.
        policy (NotationPolicy | None): Notation; ``None`` is plain text.

    Returns:
        str: The facts joined by ``", "``, or ``""`` when the object has none.
    """
    return render_facts(
        realized_facts(precond),
        _default_policy(policy),
        precision=_PRECONDITIONER_LEGEND.precision,
    )


def preconditioner_label(
    family: PreconditionerFamilyKey,
    precond: Preconditioner,
    cfg: PreconditionerConfig,
    *,
    policy: NotationPolicy | None = None,
) -> str:
    """The legend/axis display name: family abbreviation plus scientific facts.

    Args:
        family (PreconditionerFamilyKey): The base display name's source.
        precond (Preconditioner): The constructed preconditioner (realized facts).
        cfg (PreconditionerConfig): The config it was built from (configured facts).
        policy (NotationPolicy | None): Render target and symbol overrides, e.g. from
            ``[comparison_baseline.presentation]``; ``None`` is plain text.

    Returns:
        str: ``"AMG (L=2, θ=0.25, ω=0.67, c=15)"``-style text, within the legend
        budget; ``VARIANT`` facts are dropped first when it would overflow.
    """
    facts = (*configured_facts(cfg), *realized_facts(precond))
    ordered = tuple(sorted(facts, key=lambda fact: fact.importance is _VARIANT))
    description = ScientificDescription(base_name=family.abbreviation(), facts=ordered)
    return render(description, _PRECONDITIONER_LEGEND, _default_policy(policy))


def build_configured_preconditioner_label(
    cfg: PreconditionerConfig, *, policy: NotationPolicy | None = None
) -> str:
    """Render a compatibility label from config facts without rebuilding a preconditioner.

    This is used when a retained report is projected after method-major execution
    has released the prepared objects. Realized runtime facts are unavailable in
    that boundary; configured facts, family identity, and notation remain stable.
    """
    family = preconditioner_family(cfg)
    description = ScientificDescription(
        base_name=family.abbreviation(), facts=configured_facts(cfg)
    )
    return render(description, _PRECONDITIONER_LEGEND, _default_policy(policy))


def contextualize_preconditioner_label(label: str, context: str | None) -> str:
    """Append provenance context to an already-built preconditioner label.

    Keeps live preconditioner description separate from workflow provenance:
    :func:`preconditioner_label` owns the former, while composition supplies
    context such as the POD-2G fit dataset.

    Args:
        label: Base label derived from the constructed preconditioner.
        context: Optional concise provenance text.

    Returns:
        The base label followed by ``" | {context}"`` when context is present.
    """
    if context is None:
        return label
    return f"{label} | {context}"


def build_preconditioner_labels(
    preconditioners: Mapping[str, Preconditioner],
    families: Mapping[str, PreconditionerFamilyKey],
    configs: Mapping[str, PreconditionerConfig],
    *,
    policy: NotationPolicy | None = None,
) -> dict[str, str]:
    """Build a name -> display-name mapping for a set of preconditioners.

    Args:
        preconditioners: Constructed instances keyed by config name.
        families: Family key per name (see ``preconditioner_family``).
        configs: The config each was built from, per name.
        policy: Forwarded to :func:`preconditioner_label`.
    """
    return {
        name: preconditioner_label(families[name], precond, configs[name], policy=policy)
        for name, precond in preconditioners.items()
    }
