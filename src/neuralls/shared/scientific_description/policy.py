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
