"""Notation and render-profile policy for scientific descriptions.

Pure configuration types only; still standard-library only (see
``docs/plan.md``, "Rendering and notation configuration"). The Pydantic
boundary that lets TOML override these defaults is a separate, later
addition and is not required for this package to be useful and testable on
its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from neuralls.shared.scientific_description.models import FieldKey


class RenderTarget(StrEnum):
    """Output surface a rendered label is destined for."""

    PLAIN = "plain"
    """Logs and console output."""
    MATHTEXT = "mathtext"
    """Default plots; no external LaTeX dependency."""
    LATEX = "latex"
    """Publication output using LaTeX-compatible fragments."""


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """The plain and math-mode spelling of one field's symbol."""

    plain: str
    math: str


def _key(name: str) -> FieldKey:
    return FieldKey(name)


SAMPLE_COUNT = _key("sample_count")
SYSTEM_DIMENSION = _key("system_dimension")
ITERATION_COUNT = _key("iteration_count")
KRYLOV_DIMENSION = _key("krylov_dimension")
COARSE_DIMENSION = _key("coarse_dimension")
TARGET_COARSE_DIMENSION = _key("target_coarse_dimension")
MULTIGRID_LEVELS = _key("multigrid_levels")
THRESHOLD = _key("threshold")
DAMPING = _key("damping")
MEAN = _key("mean")
STANDARD_DEVIATION = _key("standard_deviation")
DROP_TOLERANCE = _key("drop_tolerance")
PRE_SMOOTHING_STEPS = _key("pre_smoothing_steps")
POST_SMOOTHING_STEPS = _key("post_smoothing_steps")
SMOOTHER_DAMPING = _key("smoother_damping")
CANDIDATE_COUNT = _key("candidate_count")
CANDIDATE_ITERATIONS = _key("candidate_iterations")
RELAXATION_TEST_VECTORS = _key("relaxation_test_vectors")
RELAXATION_SWEEPS = _key("relaxation_sweeps")
BOOTSTRAP_CYCLES = _key("bootstrap_cycles")
ALGEBRAIC_DISTANCE_THRESHOLD = _key("algebraic_distance_threshold")
COARSENING_STRATEGY = _key("coarsening_strategy")
MODEL_ARCHITECTURE = _key("model_architecture")

DEFAULT_SYMBOLS: Mapping[FieldKey, SymbolSpec] = {
    SAMPLE_COUNT: SymbolSpec(plain="N", math="N"),
    SYSTEM_DIMENSION: SymbolSpec(plain="n", math="n"),
    ITERATION_COUNT: SymbolSpec(plain="k", math="k"),
    KRYLOV_DIMENSION: SymbolSpec(plain="m", math="m"),
    COARSE_DIMENSION: SymbolSpec(plain="c", math="c"),
    TARGET_COARSE_DIMENSION: SymbolSpec(plain="c_t", math="c_t"),
    MULTIGRID_LEVELS: SymbolSpec(plain="L", math="L"),
    THRESHOLD: SymbolSpec(plain="θ", math="\\theta"),
    DAMPING: SymbolSpec(plain="ω", math="\\omega"),
    MEAN: SymbolSpec(plain="μ", math="\\mu"),
    STANDARD_DEVIATION: SymbolSpec(plain="σ", math="\\sigma"),
    DROP_TOLERANCE: SymbolSpec(plain="τ", math="\\tau"),
    PRE_SMOOTHING_STEPS: SymbolSpec(plain="ν1", math="\\nu_1"),
    POST_SMOOTHING_STEPS: SymbolSpec(plain="ν2", math="\\nu_2"),
    SMOOTHER_DAMPING: SymbolSpec(plain="ω_s", math="\\omega_s"),
    CANDIDATE_COUNT: SymbolSpec(plain="k", math="k"),
    CANDIDATE_ITERATIONS: SymbolSpec(plain="n_it", math="n_{it}"),
    RELAXATION_TEST_VECTORS: SymbolSpec(plain="k_r", math="k_r"),
    RELAXATION_SWEEPS: SymbolSpec(plain="η", math="\\eta"),
    BOOTSTRAP_CYCLES: SymbolSpec(plain="n_b", math="n_b"),
    ALGEBRAIC_DISTANCE_THRESHOLD: SymbolSpec(plain="θ_ad", math="\\theta_{ad}"),
    COARSENING_STRATEGY: SymbolSpec(plain="coarsening", math="\\mathrm{coarsening}"),
    MODEL_ARCHITECTURE: SymbolSpec(plain="arch", math="\\mathrm{arch}"),
}


@dataclass(frozen=True, slots=True)
class NotationPolicy:
    """The render target plus the symbol table used to format field keys."""

    target: RenderTarget
    symbols: Mapping[FieldKey, SymbolSpec]

    @classmethod
    def default(cls, target: RenderTarget = RenderTarget.PLAIN) -> NotationPolicy:
        """Return the policy using only the built-in default symbols."""
        return cls(target=target, symbols=DEFAULT_SYMBOLS)

    def with_overrides(self, overrides: Mapping[FieldKey, SymbolSpec]) -> NotationPolicy:
        """Return a copy with ``overrides`` merged on top of the current symbols."""
        return NotationPolicy(target=self.target, symbols={**self.symbols, **overrides})

    def symbol_for(self, key: FieldKey) -> SymbolSpec:
        """Return the symbol for ``key``, falling back to its raw name if unknown."""
        return self.symbols.get(key, SymbolSpec(plain=key.value, math=key.value))


@dataclass(frozen=True, slots=True)
class RenderProfile:
    """Controls field visibility, order, precision, and length for one render call."""

    name: str
    include_context: bool
    max_length: int | None = None
    field_order: tuple[FieldKey, ...] | None = None
    precision: Mapping[FieldKey, int] = field(default_factory=dict)


LEGEND_PROFILE = RenderProfile(name="legend", include_context=False, max_length=55)
LOG_PROFILE = RenderProfile(name="log", include_context=True)
TITLE_PROFILE = RenderProfile(name="title", include_context=True)
DETAILED_PROFILE = RenderProfile(name="detailed", include_context=True)
