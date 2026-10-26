"""Exhaustive controlled vocabulary for scientific facts.

The catalog owns semantic classification and target-specific notation. Domain
describers choose facts; renderers consume these definitions and never recover a
label from a Python field name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from neuralls.shared.scientific_description.models import FieldKey


class FactRole(StrEnum):
    """Scientific or workflow role of a controlled field."""

    ALGORITHMIC = "algorithmic"
    REPRODUCIBILITY = "reproducibility"
    SCHEDULING = "scheduling"
    OPERATIONAL = "operational"
    IDENTITY = "identity"
    PROVENANCE = "provenance"


class FactPhase(StrEnum):
    """Lifecycle phase at which a fact becomes known."""

    CONFIGURED = "configured"
    REALIZED = "realized"


class FactVisibility(StrEnum):
    """Policy used by presentation planners to select facts for a profile."""

    ALWAYS = "always"
    WHEN_DISTINGUISHING = "when_distinguishing"
    WHEN_NON_DEFAULT = "when_non_default"
    DETAILED_ONLY = "detailed_only"
    NEVER = "never"


class UnknownScientificFactError(KeyError):
    """A ``FieldKey`` has no controlled definition."""

    def __init__(self, key: FieldKey) -> None:
        self.key = key
        super().__init__(f"No scientific fact definition is registered for {key.value!r}")


class UnknownCategoricalValueError(ValueError):
    """A text fact contains a value outside its controlled vocabulary."""

    def __init__(self, key: FieldKey, value: str) -> None:
        self.key = key
        self.value = value
        super().__init__(f"Unsupported categorical value {value!r} for {key.value!r}")


class UncontrolledTextFactError(ValueError):
    """A text fact has no catalog-owned formatter or validated representation."""

    def __init__(self, key: FieldKey) -> None:
        self.key = key
        super().__init__(
            f"Text fact {key.value!r} requires a controlled formatter or validated "
            "semantic text representation"
        )


@dataclass(frozen=True, slots=True)
class CategoricalValue:
    """One semantic category and its controlled plain/MathText spellings."""

    raw: str
    plain: str
    mathtext: str


@dataclass(frozen=True, slots=True)
class CategoricalFormatter:
    """Immutable closed mapping for one categorical fact."""

    values: tuple[CategoricalValue, ...]

    def __post_init__(self) -> None:
        """Require a non-empty mapping with one row per raw value."""
        raw_values = tuple(value.raw for value in self.values)
        if not self.values or len(raw_values) != len(set(raw_values)):
            raise ValueError("CategoricalFormatter values must be non-empty and unique")

    def plain(self, key: FieldKey, value: str) -> str:
        """Return the controlled plain spelling or reject ``value``."""
        return self._find(key, value).plain

    def mathtext(self, key: FieldKey, value: str) -> str:
        """Return the controlled MathText spelling or reject ``value``."""
        return self._find(key, value).mathtext

    def _find(self, key: FieldKey, value: str) -> CategoricalValue:
        for candidate in self.values:
            if candidate.raw == value:
                return candidate
        raise UnknownCategoricalValueError(key, value)


@dataclass(frozen=True, slots=True)
class FactDefinition:
    """One controlled fact's semantics, order, notation, and visibility."""

    key: FieldKey
    role: FactRole
    phase: FactPhase
    order: int
    precision: int | None
    plain_symbol: str
    mathtext_symbol: str
    categorical_formatter: CategoricalFormatter | None
    visibility_policy: FactVisibility

    def __post_init__(self) -> None:
        """Validate ordering, precision, and notation invariants."""
        if self.order < 0:
            raise ValueError("FactDefinition.order must be non-negative")
        if self.precision is not None and self.precision < 1:
            raise ValueError("FactDefinition.precision must be positive")
        if not self.plain_symbol or not self.mathtext_symbol:
            raise ValueError("FactDefinition symbols must not be empty")


def _key(name: str) -> FieldKey:
    return FieldKey(name)


SAMPLE_COUNT = _key("sample_count")
SYSTEM_DIMENSION = _key("system_dimension")
ITERATION_COUNT = _key("iteration_count")
KRYLOV_DIMENSION = _key("krylov_dimension")
COARSE_DIMENSION = _key("coarse_dimension")
TARGET_COARSE_DIMENSION = _key("target_coarse_dimension")
MAX_COARSE_DIMENSION = _key("max_coarse_dimension")
MULTIGRID_LEVELS = _key("multigrid_levels")
THRESHOLD = _key("threshold")
DAMPING = _key("damping")
MEAN = _key("mean")
STANDARD_DEVIATION = _key("standard_deviation")
LOWER_BOUND = _key("lower_bound")
UPPER_BOUND = _key("upper_bound")
DROP_TOLERANCE = _key("drop_tolerance")
PRE_SMOOTHING_STEPS = _key("pre_smoothing_steps")
POST_SMOOTHING_STEPS = _key("post_smoothing_steps")
SMOOTHER_DAMPING = _key("smoother_damping")
CANDIDATE_COUNT = _key("candidate_count")
CANDIDATE_ITERATIONS = _key("candidate_iterations")
RELAXATION_TEST_VECTORS = _key("relaxation_test_vectors")
RELAXATION_SWEEPS = _key("relaxation_sweeps")
BOOTSTRAP_CYCLES = _key("bootstrap_cycles")
COMPATIBLE_RELAXATION_SWEEPS = _key("compatible_relaxation_sweeps")
COMPATIBLE_RELAXATION_TOLERANCE = _key("compatible_relaxation_tolerance")
ALGEBRAIC_DISTANCE_THRESHOLD = _key("algebraic_distance_threshold")
CALIBER = _key("caliber")
CALIBER_GROWTH_PENALTY = _key("caliber_growth_penalty")
USE_LSR = _key("use_lsr")
RANK = _key("rank")
SNAPSHOT_COUNT = _key("snapshots")
WEIGHTING = _key("weighting")
METRIC = _key("metric")
BETA = _key("beta")
STEPS = _key("steps")
PROBE_DISTRIBUTION = _key("probe_distribution")
COARSENING_STRATEGY = _key("coarsening_strategy")
MODEL_ARCHITECTURE = _key("model_architecture")
ASSIGNMENT = _key("assignment")
SEED = _key("seed")


def _categories(*values: tuple[str, str, str]) -> CategoricalFormatter:
    return CategoricalFormatter(tuple(CategoricalValue(*value) for value in values))


def _definition(
    key: FieldKey,
    role: FactRole,
    phase: FactPhase,
    order: int,
    precision: int | None,
    plain_symbol: str,
    mathtext_symbol: str,
    *,
    categories: CategoricalFormatter | None = None,
    visibility: FactVisibility = FactVisibility.ALWAYS,
) -> FactDefinition:
    return FactDefinition(
        key=key,
        role=role,
        phase=phase,
        order=order,
        precision=precision,
        plain_symbol=plain_symbol,
        mathtext_symbol=mathtext_symbol,
        categorical_formatter=categories,
        visibility_policy=visibility,
    )


_A = FactRole.ALGORITHMIC
_C = FactPhase.CONFIGURED
_R = FactPhase.REALIZED

FACT_DEFINITIONS: tuple[FactDefinition, ...] = (
    _definition(SAMPLE_COUNT, _A, _R, 10, None, "N", "N"),
    _definition(
        SYSTEM_DIMENSION,
        FactRole.PROVENANCE,
        _R,
        20,
        None,
        "n",
        "n",
        visibility=FactVisibility.DETAILED_ONLY,
    ),
    _definition(ITERATION_COUNT, _A, _C, 30, None, "k", "k"),
    _definition(KRYLOV_DIMENSION, _A, _C, 40, None, "m", "m"),
    _definition(COARSE_DIMENSION, _A, _R, 50, None, "c", "c"),
    _definition(TARGET_COARSE_DIMENSION, _A, _C, 60, None, "cₜ", "c_t"),
    _definition(MAX_COARSE_DIMENSION, _A, _C, 70, None, "cₘₐₓ", "c_{\\max}"),
    _definition(MULTIGRID_LEVELS, _A, _C, 80, None, "L", "L"),
    _definition(THRESHOLD, _A, _C, 90, 3, "θ", "\\theta"),
    _definition(
        DAMPING,
        _A,
        _C,
        100,
        3,
        "ω",
        "\\omega",
        categories=_categories(("auto", "auto", "\\mathrm{auto}")),
    ),
    _definition(MEAN, _A, _C, 110, 3, "μ", "\\mu"),
    _definition(STANDARD_DEVIATION, _A, _C, 120, 3, "σ", "\\sigma"),
    _definition(LOWER_BOUND, _A, _C, 130, 3, "a", "a"),
    _definition(UPPER_BOUND, _A, _C, 140, 3, "b", "b"),
    _definition(DROP_TOLERANCE, _A, _C, 150, 1, "τ", "\\tau"),
    _definition(PRE_SMOOTHING_STEPS, _A, _C, 160, None, "ν₁", "\\nu_1"),
    _definition(POST_SMOOTHING_STEPS, _A, _C, 170, None, "ν₂", "\\nu_2"),
    _definition(
        SMOOTHER_DAMPING,
        _A,
        _C,
        180,
        3,
        "ωₛ",
        "\\omega_s",
        categories=_categories(("auto", "auto", "\\mathrm{auto}")),
    ),
    _definition(CANDIDATE_COUNT, _A, _C, 190, None, "k", "k"),
    _definition(CANDIDATE_ITERATIONS, _A, _C, 200, None, "nᵢₜ", "n_{\\mathrm{it}}"),
    _definition(RELAXATION_TEST_VECTORS, _A, _C, 210, None, "kᵣ", "k_r"),
    _definition(RELAXATION_SWEEPS, _A, _C, 220, None, "η", "\\eta"),
    _definition(BOOTSTRAP_CYCLES, _A, _C, 230, None, "nᵦ", "n_b"),
    _definition(COMPATIBLE_RELAXATION_SWEEPS, _A, _C, 240, None, "ν", "\\nu"),
    _definition(COMPATIBLE_RELAXATION_TOLERANCE, _A, _C, 250, 3, "δ", "\\delta"),
    _definition(
        ALGEBRAIC_DISTANCE_THRESHOLD,
        _A,
        _C,
        260,
        3,
        "θₐd",
        "\\theta_{\\mathrm{ad}}",
    ),
    _definition(CALIBER, _A, _C, 270, None, "q", "q"),
    _definition(CALIBER_GROWTH_PENALTY, _A, _C, 280, 3, "γ", "\\gamma"),
    _definition(
        USE_LSR,
        _A,
        _C,
        290,
        None,
        "LSR",
        "\\mathrm{LSR}",
        categories=_categories(("off", "off", "\\mathrm{off}"), ("on", "on", "\\mathrm{on}")),
    ),
    _definition(RANK, _A, _C, 300, 4, "r", "r"),
    _definition(SNAPSHOT_COUNT, _A, _C, 310, None, "Nₛ", "N_s"),
    _definition(
        WEIGHTING,
        _A,
        _C,
        320,
        None,
        "w",
        "w",
        categories=_categories(
            ("power_norm", "PN", "\\mathrm{PN}"),
            ("raw", "raw", "\\mathrm{raw}"),
            ("smoother_persistence", "SP", "\\mathrm{SP}"),
            ("uniform", "uniform", "\\mathrm{uniform}"),
        ),
    ),
    _definition(
        METRIC,
        _A,
        _C,
        330,
        None,
        "p",
        "p",
        categories=_categories(("a", "A", "A"), ("l2", "ℓ₂", "\\ell_2")),
    ),
    _definition(BETA, _A, _C, 340, 3, "β", "\\beta"),
    _definition(STEPS, _A, _C, 350, None, "ν", "\\nu"),
    _definition(
        PROBE_DISTRIBUTION,
        _A,
        _C,
        360,
        None,
        "p",
        "p",
        categories=_categories(
            ("gaussian", "𝒩", "\\mathcal{N}"),
            ("rademacher", "Rad", "\\mathrm{Rad}"),
        ),
    ),
    _definition(
        COARSENING_STRATEGY,
        _A,
        _C,
        370,
        None,
        "C",
        "C",
        categories=_categories(
            ("aggregation", "SA", "\\mathrm{SA}"),
            ("neural_pod", "Neural-POD", "\\mathrm{Neural-POD}"),
            ("pod", "POD", "\\mathrm{POD}"),
            ("target_dim", "target-c", "\\mathrm{target-c}"),
        ),
    ),
    _definition(MODEL_ARCHITECTURE, _A, _R, 380, None, "arch", "\\mathrm{arch}"),
    _definition(
        ASSIGNMENT,
        FactRole.IDENTITY,
        _C,
        390,
        None,
        "id",
        "\\mathrm{id}",
        visibility=FactVisibility.NEVER,
    ),
    _definition(
        SEED,
        FactRole.REPRODUCIBILITY,
        _C,
        400,
        None,
        "s",
        "s",
        visibility=FactVisibility.WHEN_DISTINGUISHING,
    ),
)

_FACTS_BY_KEY: Mapping[FieldKey, FactDefinition] = MappingProxyType(
    {definition.key: definition for definition in FACT_DEFINITIONS}
)


def fact_definition(key: FieldKey) -> FactDefinition:
    """Return ``key``'s definition without falling back to its raw name."""
    try:
        return _FACTS_BY_KEY[key]
    except KeyError as error:
        raise UnknownScientificFactError(key) from error
