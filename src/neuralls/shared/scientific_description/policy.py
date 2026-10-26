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
from types import MappingProxyType

from neuralls.shared.scientific_description.catalog import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BOOTSTRAP_CYCLES,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COARSE_DIMENSION,
    COARSENING_STRATEGY,
    DAMPING,
    DROP_TOLERANCE,
    FACT_DEFINITIONS,
    ITERATION_COUNT,
    KRYLOV_DIMENSION,
    MEAN,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SAMPLE_COUNT,
    SMOOTHER_DAMPING,
    STANDARD_DEVIATION,
    SYSTEM_DIMENSION,
    TARGET_COARSE_DIMENSION,
    THRESHOLD,
    fact_definition,
)
from neuralls.shared.scientific_description.models import FieldKey

__all__ = [
    "ALGEBRAIC_DISTANCE_THRESHOLD",
    "BOOTSTRAP_CYCLES",
    "CANDIDATE_COUNT",
    "CANDIDATE_ITERATIONS",
    "COARSENING_STRATEGY",
    "COARSE_DIMENSION",
    "DAMPING",
    "DEFAULT_SYMBOLS",
    "DETAILED_PROFILE",
    "DROP_TOLERANCE",
    "ITERATION_COUNT",
    "KRYLOV_DIMENSION",
    "LEGEND_PROFILE",
    "LOG_PROFILE",
    "MEAN",
    "MODEL_ARCHITECTURE",
    "MULTIGRID_LEVELS",
    "POST_SMOOTHING_STEPS",
    "PRE_SMOOTHING_STEPS",
    "RELAXATION_SWEEPS",
    "RELAXATION_TEST_VECTORS",
    "SAMPLE_COUNT",
    "SMOOTHER_DAMPING",
    "STANDARD_DEVIATION",
    "SYSTEM_DIMENSION",
    "TARGET_COARSE_DIMENSION",
    "THRESHOLD",
    "TITLE_PROFILE",
    "NotationPolicy",
    "RenderProfile",
    "RenderTarget",
    "SymbolSpec",
]


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


DEFAULT_SYMBOLS: Mapping[FieldKey, SymbolSpec] = MappingProxyType(
    {
        definition.key: SymbolSpec(
            plain=definition.plain_symbol,
            math=definition.mathtext_symbol,
        )
        for definition in FACT_DEFINITIONS
    }
)


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
        for key in overrides:
            fact_definition(key)
        return NotationPolicy(target=self.target, symbols={**self.symbols, **overrides})

    def symbol_for(self, key: FieldKey) -> SymbolSpec:
        """Return the controlled symbol for ``key`` or reject an unknown key."""
        definition = fact_definition(key)
        return self.symbols.get(
            key,
            SymbolSpec(plain=definition.plain_symbol, math=definition.mathtext_symbol),
        )


@dataclass(frozen=True, slots=True)
class RenderProfile:
    """Controls field visibility, order, precision, and preferred line length."""

    name: str
    include_context: bool
    max_length: int | None = None
    field_order: tuple[FieldKey, ...] | None = None
    precision: Mapping[FieldKey, int] = field(default_factory=dict)


LEGEND_PROFILE = RenderProfile(name="legend", include_context=False, max_length=55)
LOG_PROFILE = RenderProfile(name="log", include_context=True)
TITLE_PROFILE = RenderProfile(name="title", include_context=True)
DETAILED_PROFILE = RenderProfile(name="detailed", include_context=True)
