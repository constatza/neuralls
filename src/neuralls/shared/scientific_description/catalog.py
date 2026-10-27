"""Exhaustive controlled vocabulary for scientific facts.

The catalog owns semantic classification and target-specific notation. Domain
describers choose facts; renderers consume these definitions and never recover a
label from a Python field name.

The catalog is one module, grouped into named sections by the owner that decides the
facts (generation strategies, preconditioners, scheduling), with the keys several owners
share first. It is not registered piecemeal from the owner packages: the planner, the
renderer, and the manifest codec all need every definition, and a registry filled by
import side effects would let a process that never imported an owner (for example one that
only decodes a dataset manifest) fail to know that owner's keys. The catalog validates
itself at import (unique keys and orders, declared shared symbols), so a new fact is added
here once and checked immediately.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import assert_never

from neuralls.shared.scientific_description.models import (
    DecimalFact,
    DisplayTextFact,
    FactValueKind,
    FieldKey,
    IntegerFact,
    ScientificFact,
    TextFact,
    ValidatedLabel,
)


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


def is_compact_by_policy(visibility: FactVisibility) -> bool:
    """Whether a policy puts a fact in compact labels without comparing it to other methods.

    ``WHEN_DISTINGUISHING`` is compact only when the value differs between peers, which
    only a cohort planner can decide, so it is not compact by policy alone. This is the one
    place the compact decision for the other policies is made.
    """
    match visibility:
        case FactVisibility.ALWAYS | FactVisibility.WHEN_NON_DEFAULT:
            return True
        case (
            FactVisibility.WHEN_DISTINGUISHING | FactVisibility.DETAILED_ONLY | FactVisibility.NEVER
        ):
            return False
        case _ as unreachable:
            assert_never(unreachable)


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


class FactValueKindError(ValueError):
    """A fact's value is of a kind its catalog definition does not allow."""

    def __init__(
        self, key: FieldKey, got: FactValueKind, allowed: frozenset[FactValueKind]
    ) -> None:
        self.key = key
        self.got = got
        names = ", ".join(sorted(kind.value for kind in allowed))
        super().__init__(f"Fact {key.value!r} takes a {names} value, not {got.value}")


class CatalogError(ValueError):
    """The catalog's own definitions are inconsistent."""


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
    """One controlled fact's semantics, order, notation, value kinds, and visibility."""

    key: FieldKey
    role: FactRole
    phase: FactPhase
    order: int
    precision: int | None
    plain_symbol: str
    mathtext_symbol: str
    value_kinds: frozenset[FactValueKind]
    categorical_formatter: CategoricalFormatter | None
    visibility_policy: FactVisibility

    def __post_init__(self) -> None:
        """Validate ordering, precision, notation, and value-kind invariants."""
        if self.order < 0:
            raise ValueError("FactDefinition.order must be non-negative")
        if self.precision is not None and self.precision < 1:
            raise ValueError("FactDefinition.precision must be positive")
        if not self.plain_symbol or not self.mathtext_symbol:
            raise ValueError("FactDefinition symbols must not be empty")
        if not self.value_kinds:
            raise ValueError("FactDefinition.value_kinds must not be empty")
        if (FactValueKind.TEXT in self.value_kinds) != (self.categorical_formatter is not None):
            raise ValueError(
                f"{self.key.value!r}: a text value needs categories, and categories only apply "
                "to a text value"
            )
        if FactValueKind.DECIMAL in self.value_kinds and self.precision is None:
            raise ValueError(f"{self.key.value!r}: a decimal fact needs a precision")


def _key(name: str) -> FieldKey:
    return FieldKey(name)


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
    value_kinds: frozenset[FactValueKind],
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
        value_kinds=value_kinds,
        categorical_formatter=categories,
        visibility_policy=visibility,
    )


_A = FactRole.ALGORITHMIC
_C = FactPhase.CONFIGURED
_R = FactPhase.REALIZED

# ==============================================================================
# Facts several owners share
# Counts and seeds that datasets, methods, and realized systems all report, and the two keys (damping, steps) whose symbol means the same thing for a strategy and a preconditioner.
# ==============================================================================
SAMPLE_COUNT = _key("sample_count")
SYSTEM_DIMENSION = _key("system_dimension")
ITERATION_COUNT = _key("iteration_count")
SEED = _key("seed")
DAMPING = _key("damping")
STEPS = _key("steps")

# ==============================================================================
# Generation-strategy facts
# Owned by the generation-strategy describers (domain/generation/descriptions.py).
# ==============================================================================
KRYLOV_DIMENSION = _key("krylov_dimension")
MEAN = _key("mean")
STANDARD_DEVIATION = _key("standard_deviation")
LOWER_BOUND = _key("lower_bound")
UPPER_BOUND = _key("upper_bound")
PROBE_DISTRIBUTION = _key("probe_distribution")
WINDOW_START = _key("window_start")
WINDOW_STRIDE = _key("window_stride")
EIGENVALUE_SELECTION = _key("eigenvalue_selection")
EIGENVECTOR_COUNT = _key("eigenvector_count")
EIGENVECTOR_INCLUSION = _key("eigenvector_inclusion")
SOLVER_KIND = _key("solver_kind")
SOLVER_RELATIVE_TOLERANCE = _key("solver_relative_tolerance")
SOLVER_ABSOLUTE_TOLERANCE = _key("solver_absolute_tolerance")
SOLVER_MAX_ITERATIONS = _key("solver_max_iterations")
SOLVER_POSITIVE_DEFINITE = _key("solver_positive_definite")
CONSTANT_VALUE = _key("constant_value")
FREQUENCY = _key("frequency")
RHS_SCALE = _key("rhs_scale")
SOLUTION_SCALE = _key("solution_scale")
VERIFICATION_TOLERANCE = _key("verification_tolerance")
SPARSE_SUPPORT = _key("sparse_support")
MASK_DENSITY = _key("mask_density")

# ==============================================================================
# Preconditioner facts
# Owned by the preconditioner describers (composition/preconditioners/descriptions.py).
# ==============================================================================
COARSE_DIMENSION = _key("coarse_dimension")
TARGET_COARSE_DIMENSION = _key("target_coarse_dimension")
MAX_COARSE_DIMENSION = _key("max_coarse_dimension")
MULTIGRID_LEVELS = _key("multigrid_levels")
THRESHOLD = _key("threshold")
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
SNAPSHOT_COUNT = _key("snapshot_count")
WEIGHTING = _key("weighting")
METRIC = _key("metric")
BETA = _key("beta")
MODEL_ARCHITECTURE = _key("model_architecture")
REALIZED_LEVELS = _key("realized_levels")

# ==============================================================================
# Scheduling and identity facts
# Scheduling controls change the executed method; the assignment key is identity and never shown.
# ==============================================================================
ASSIGNMENT = _key("assignment")
START_ITERATION = _key("start_iteration")
ITERATION_LIMIT = _key("iteration_limit")
FALLBACK = _key("fallback")


FACT_DEFINITIONS: tuple[FactDefinition, ...] = (
    # --- Facts several owners share ---
    _definition(SAMPLE_COUNT, _A, _R, 10, None, "N", "N", frozenset({FactValueKind.INTEGER})),
    _definition(
        SYSTEM_DIMENSION,
        FactRole.PROVENANCE,
        _R,
        20,
        None,
        "n",
        "n",
        frozenset({FactValueKind.INTEGER}),
        visibility=FactVisibility.DETAILED_ONLY,
    ),
    _definition(ITERATION_COUNT, _A, _C, 30, None, "k", "k", frozenset({FactValueKind.INTEGER})),
    _definition(
        SEED,
        FactRole.REPRODUCIBILITY,
        _C,
        400,
        None,
        "s",
        "s",
        frozenset({FactValueKind.INTEGER}),
        visibility=FactVisibility.WHEN_DISTINGUISHING,
    ),
    _definition(
        DAMPING,
        _A,
        _C,
        100,
        3,
        "ω",
        "\\omega",
        frozenset({FactValueKind.TEXT, FactValueKind.DECIMAL}),
        categories=_categories(("auto", "auto", "\\mathrm{auto}")),
    ),
    _definition(STEPS, _A, _C, 350, None, "ν", "\\nu", frozenset({FactValueKind.INTEGER})),
    # --- Generation-strategy facts ---
    _definition(KRYLOV_DIMENSION, _A, _C, 40, None, "m", "m", frozenset({FactValueKind.INTEGER})),
    _definition(MEAN, _A, _C, 110, 3, "μ", "\\mu", frozenset({FactValueKind.DECIMAL})),
    _definition(
        STANDARD_DEVIATION, _A, _C, 120, 3, "σ", "\\sigma", frozenset({FactValueKind.DECIMAL})
    ),
    _definition(LOWER_BOUND, _A, _C, 130, 3, "a", "a", frozenset({FactValueKind.DECIMAL})),
    _definition(UPPER_BOUND, _A, _C, 140, 3, "b", "b", frozenset({FactValueKind.DECIMAL})),
    _definition(
        PROBE_DISTRIBUTION,
        _A,
        _C,
        360,
        None,
        "p",
        "p",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(
            ("gaussian", "𝒩", "\\mathcal{N}"), ("rademacher", "Rad", "\\mathrm{Rad}")
        ),
    ),
    _definition(WINDOW_START, _A, _C, 450, None, "kₛ", "k_s", frozenset({FactValueKind.INTEGER})),
    _definition(
        WINDOW_STRIDE, _A, _C, 460, None, "Δk", "\\Delta k", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        EIGENVALUE_SELECTION,
        _A,
        _C,
        470,
        None,
        "λ",
        "\\lambda",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(
            ("smallest", "min", "\\mathrm{min}"),
            ("largest", "max", "\\mathrm{max}"),
            ("random", "rand", "\\mathrm{rand}"),
        ),
    ),
    _definition(
        EIGENVECTOR_COUNT, _A, _C, 480, None, "nₑ", "n_e", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        EIGENVECTOR_INCLUSION,
        _A,
        _C,
        490,
        None,
        "EV",
        "\\mathrm{EV}",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(("off", "off", "\\mathrm{off}"), ("on", "on", "\\mathrm{on}")),
    ),
    _definition(
        SOLVER_KIND,
        _A,
        _C,
        500,
        None,
        "S",
        "S",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(
            ("direct", "direct", "\\mathrm{direct}"), ("cg", "CG", "\\mathrm{CG}")
        ),
    ),
    _definition(
        SOLVER_RELATIVE_TOLERANCE,
        _A,
        _C,
        510,
        3,
        "εᵣ",
        "\\epsilon_r",
        frozenset({FactValueKind.DECIMAL}),
    ),
    _definition(
        SOLVER_ABSOLUTE_TOLERANCE,
        _A,
        _C,
        520,
        3,
        "εₐ",
        "\\epsilon_a",
        frozenset({FactValueKind.DECIMAL}),
    ),
    _definition(
        SOLVER_MAX_ITERATIONS,
        _A,
        _C,
        530,
        None,
        "Kₘₐₓ",
        "K_{\\max}",
        frozenset({FactValueKind.INTEGER}),
    ),
    _definition(
        SOLVER_POSITIVE_DEFINITE,
        _A,
        _C,
        540,
        None,
        "SPD",
        "\\mathrm{SPD}",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(("off", "off", "\\mathrm{off}"), ("on", "on", "\\mathrm{on}")),
    ),
    _definition(CONSTANT_VALUE, _A, _C, 550, 3, "v", "v", frozenset({FactValueKind.DECIMAL})),
    _definition(FREQUENCY, _A, _C, 560, 3, "ω", "\\omega", frozenset({FactValueKind.DECIMAL})),
    _definition(
        RHS_SCALE, _A, _C, 570, 3, "‖b‖", "\\Vert b\\Vert", frozenset({FactValueKind.DECIMAL})
    ),
    _definition(SOLUTION_SCALE, _A, _C, 580, 3, "α", "\\alpha", frozenset({FactValueKind.DECIMAL})),
    _definition(
        VERIFICATION_TOLERANCE,
        _A,
        _C,
        590,
        3,
        "εᵥ",
        "\\epsilon_v",
        frozenset({FactValueKind.DECIMAL}),
    ),
    _definition(
        SPARSE_SUPPORT,
        _A,
        _C,
        600,
        None,
        "nnz",
        "\\mathrm{nnz}",
        frozenset({FactValueKind.INTEGER}),
    ),
    _definition(MASK_DENSITY, _A, _C, 610, 3, "ρ", "\\rho", frozenset({FactValueKind.DECIMAL})),
    # --- Preconditioner facts ---
    _definition(COARSE_DIMENSION, _A, _R, 50, None, "c", "c", frozenset({FactValueKind.INTEGER})),
    _definition(
        TARGET_COARSE_DIMENSION, _A, _C, 60, None, "cₜ", "c_t", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        MAX_COARSE_DIMENSION,
        _A,
        _C,
        70,
        None,
        "cₘₐₓ",
        "c_{\\max}",
        frozenset({FactValueKind.INTEGER}),
    ),
    _definition(MULTIGRID_LEVELS, _A, _C, 80, None, "L", "L", frozenset({FactValueKind.INTEGER})),
    _definition(THRESHOLD, _A, _C, 90, 3, "θ", "\\theta", frozenset({FactValueKind.DECIMAL})),
    _definition(DROP_TOLERANCE, _A, _C, 150, 1, "τ", "\\tau", frozenset({FactValueKind.DECIMAL})),
    _definition(
        PRE_SMOOTHING_STEPS, _A, _C, 160, None, "ν₁", "\\nu_1", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        POST_SMOOTHING_STEPS, _A, _C, 170, None, "ν₂", "\\nu_2", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        SMOOTHER_DAMPING,
        _A,
        _C,
        180,
        3,
        "ωₛ",
        "\\omega_s",
        frozenset({FactValueKind.TEXT, FactValueKind.DECIMAL}),
        categories=_categories(("auto", "auto", "\\mathrm{auto}")),
    ),
    _definition(CANDIDATE_COUNT, _A, _C, 190, None, "k", "k", frozenset({FactValueKind.INTEGER})),
    _definition(
        CANDIDATE_ITERATIONS,
        _A,
        _C,
        200,
        None,
        "nᵢₜ",
        "n_{\\mathrm{it}}",
        frozenset({FactValueKind.INTEGER}),
    ),
    _definition(
        RELAXATION_TEST_VECTORS, _A, _C, 210, None, "kᵣ", "k_r", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        RELAXATION_SWEEPS, _A, _C, 220, None, "η", "\\eta", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        BOOTSTRAP_CYCLES, _A, _C, 230, None, "nᵦ", "n_b", frozenset({FactValueKind.INTEGER})
    ),
    _definition(
        COMPATIBLE_RELAXATION_SWEEPS,
        _A,
        _C,
        240,
        None,
        "ν",
        "\\nu",
        frozenset({FactValueKind.INTEGER}),
    ),
    _definition(
        COMPATIBLE_RELAXATION_TOLERANCE,
        _A,
        _C,
        250,
        3,
        "δ",
        "\\delta",
        frozenset({FactValueKind.DECIMAL}),
    ),
    _definition(
        ALGEBRAIC_DISTANCE_THRESHOLD,
        _A,
        _C,
        260,
        3,
        "θₐd",
        "\\theta_{\\mathrm{ad}}",
        frozenset({FactValueKind.DECIMAL}),
    ),
    _definition(CALIBER, _A, _C, 270, None, "q", "q", frozenset({FactValueKind.INTEGER})),
    _definition(
        CALIBER_GROWTH_PENALTY, _A, _C, 280, 3, "γ", "\\gamma", frozenset({FactValueKind.DECIMAL})
    ),
    _definition(
        USE_LSR,
        _A,
        _C,
        290,
        None,
        "LSR",
        "\\mathrm{LSR}",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(("off", "off", "\\mathrm{off}"), ("on", "on", "\\mathrm{on}")),
    ),
    _definition(
        RANK, _A, _C, 300, 4, "r", "r", frozenset({FactValueKind.INTEGER, FactValueKind.DECIMAL})
    ),
    _definition(SNAPSHOT_COUNT, _A, _C, 310, None, "Nₛ", "N_s", frozenset({FactValueKind.INTEGER})),
    _definition(
        WEIGHTING,
        _A,
        _C,
        320,
        None,
        "w",
        "w",
        frozenset({FactValueKind.TEXT}),
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
        frozenset({FactValueKind.TEXT}),
        categories=_categories(("a", "A", "A"), ("l2", "ℓ₂", "\\ell_2")),
    ),
    _definition(BETA, _A, _C, 340, 3, "β", "\\beta", frozenset({FactValueKind.DECIMAL})),
    _definition(
        MODEL_ARCHITECTURE,
        _A,
        _C,
        380,
        None,
        "arch",
        "\\mathrm{arch}",
        frozenset({FactValueKind.LABEL}),
    ),
    _definition(
        REALIZED_LEVELS, _A, _R, 440, None, "Lᵣ", "L_r", frozenset({FactValueKind.INTEGER})
    ),
    # --- Scheduling and identity facts ---
    _definition(
        ASSIGNMENT,
        FactRole.IDENTITY,
        _C,
        390,
        None,
        "id",
        "\\mathrm{id}",
        frozenset({FactValueKind.INTEGER}),
        visibility=FactVisibility.NEVER,
    ),
    _definition(
        START_ITERATION,
        FactRole.SCHEDULING,
        _C,
        410,
        None,
        "k₀",
        "k_0",
        frozenset({FactValueKind.INTEGER}),
        visibility=FactVisibility.WHEN_NON_DEFAULT,
    ),
    _definition(
        ITERATION_LIMIT,
        FactRole.SCHEDULING,
        _C,
        420,
        None,
        "kₘₐₓ",
        "k_{\\max}",
        frozenset({FactValueKind.INTEGER}),
        visibility=FactVisibility.WHEN_NON_DEFAULT,
    ),
    _definition(
        FALLBACK,
        FactRole.SCHEDULING,
        _C,
        430,
        None,
        "fb",
        "\\mathrm{fb}",
        frozenset({FactValueKind.TEXT}),
        categories=_categories(
            ("identity", "Identity", "\\mathrm{Identity}"),
            ("jacobi", "Jacobi", "\\mathrm{Jacobi}"),
            ("ilu", "ILU", "\\mathrm{ILU}"),
            ("ic0", "IC(0)", "\\mathrm{IC(0)}"),
            ("neural", "Neural", "\\mathrm{Neural}"),
            ("amg", "AMG", "\\mathrm{AMG}"),
            ("neural_amg", "Neural AMG", "\\mathrm{Neural\\ AMG}"),
            ("adaptive_sa_amg", "aSA", "\\mathrm{aSA}"),
            ("bootstrap_amg", "BAMG", "\\mathrm{BAMG}"),
        ),
        visibility=FactVisibility.WHEN_NON_DEFAULT,
    ),
)

SHARED_SYMBOL_GROUPS: frozenset[frozenset[FieldKey]] = frozenset(
    {
        frozenset({ITERATION_COUNT, CANDIDATE_COUNT}),
        frozenset({DAMPING, FREQUENCY}),
        frozenset({COMPATIBLE_RELAXATION_SWEEPS, STEPS}),
        frozenset({METRIC, PROBE_DISTRIBUTION}),
    }
)
"""Keys that deliberately share one plain symbol because they never describe the same object
(for example the iteration count of a residual trace and the candidate count of an adaptive
hierarchy). Any other repeated symbol is a catalog error."""


def validate_catalog(
    definitions: Iterable[FactDefinition], shared_symbols: frozenset[frozenset[FieldKey]]
) -> Mapping[FieldKey, FactDefinition]:
    """Return the definitions by key, or raise ``CatalogError`` if they are inconsistent."""
    by_key: dict[FieldKey, FactDefinition] = {}
    by_order: dict[int, FieldKey] = {}
    by_symbol: dict[str, set[FieldKey]] = {}
    for definition in definitions:
        if definition.key in by_key:
            raise CatalogError(f"duplicate fact key {definition.key.value!r}")
        if definition.order in by_order:
            raise CatalogError(
                f"fact keys {by_order[definition.order].value!r} and {definition.key.value!r} "
                f"share order {definition.order}"
            )
        by_key[definition.key] = definition
        by_order[definition.order] = definition.key
        by_symbol.setdefault(definition.plain_symbol, set()).add(definition.key)
    colliding = {frozenset(keys) for keys in by_symbol.values() if len(keys) > 1}
    undeclared = colliding - shared_symbols
    if undeclared:
        names = sorted(sorted(key.value for key in group) for group in undeclared)
        raise CatalogError(f"plain symbol shared without being declared: {names}")
    stale = shared_symbols - colliding
    if stale:
        names = sorted(sorted(key.value for key in group) for group in stale)
        raise CatalogError(f"declared shared symbol that no longer collides: {names}")
    return MappingProxyType(by_key)


_FACTS_BY_KEY: Mapping[FieldKey, FactDefinition] = validate_catalog(
    FACT_DEFINITIONS, SHARED_SYMBOL_GROUPS
)


def fact_definition(key: FieldKey) -> FactDefinition:
    """Return ``key``'s definition without falling back to its raw name."""
    try:
        return _FACTS_BY_KEY[key]
    except KeyError as error:
        raise UnknownScientificFactError(key) from error


def make_fact(key: FieldKey, value: float | str | ValidatedLabel) -> ScientificFact:
    """Build the fact for ``key``, checking the key and the value's kind against the catalog.

    A Python ``bool`` is never accepted. An ``int`` and a ``float`` are distinct kinds, so a
    decimal-only key rejects ``3`` just as an integer-only key rejects ``3.0``. A ``str`` must
    be one of the key's controlled categories; a ``ValidatedLabel`` is only for label keys.

    Raises:
        UnknownScientificFactError: ``key`` is not in the catalog.
        FactValueKindError: ``value`` is of a kind the key does not take.
        UnknownCategoricalValueError: ``value`` is not one of the key's categories.
    """
    definition = fact_definition(key)
    match value:
        case bool():
            raise TypeError(f"Fact {key.value!r} cannot take a bool")
        case int():
            _require_kind(definition, FactValueKind.INTEGER)
            return IntegerFact(key, value)
        case float():
            _require_kind(definition, FactValueKind.DECIMAL)
            return DecimalFact(key, value)
        case str():
            _require_kind(definition, FactValueKind.TEXT)
            _category_formatter(definition).plain(key, value)
            return TextFact(key, value)
        case ValidatedLabel():
            _require_kind(definition, FactValueKind.LABEL)
            return DisplayTextFact(key, value)
        case _:
            raise TypeError(f"Fact {key.value!r} cannot take a {type(value).__name__} value")


def _require_kind(definition: FactDefinition, kind: FactValueKind) -> None:
    if kind not in definition.value_kinds:
        raise FactValueKindError(definition.key, kind, definition.value_kinds)


def _category_formatter(definition: FactDefinition) -> CategoricalFormatter:
    formatter = definition.categorical_formatter
    if formatter is None:
        raise UncontrolledTextFactError(definition.key)
    return formatter
