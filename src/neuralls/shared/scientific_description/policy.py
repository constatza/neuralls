"""Notation and render-profile policy for scientific descriptions.

Pure configuration types only. A Pydantic boundary that lets TOML override
the catalog's symbols is a separate concern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from neuralls.shared.scientific_description.catalog import fact_definition
from neuralls.shared.scientific_description.models import FieldKey

_NO_OVERRIDES: Mapping[FieldKey, SymbolSpec] = MappingProxyType({})


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


@dataclass(frozen=True, slots=True)
class NotationPolicy:
    """The render target plus the symbols that replace the catalog's defaults."""

    target: RenderTarget
    overrides: Mapping[FieldKey, SymbolSpec] = _NO_OVERRIDES

    @classmethod
    def default(cls, target: RenderTarget = RenderTarget.PLAIN) -> NotationPolicy:
        """Return the policy using only the catalog's symbols."""
        return cls(target=target)

    def with_overrides(self, overrides: Mapping[FieldKey, SymbolSpec]) -> NotationPolicy:
        """Return a copy with ``overrides`` merged on top of the current overrides."""
        for key in overrides:
            fact_definition(key)
        return NotationPolicy(target=self.target, overrides={**self.overrides, **overrides})

    def symbol_for(self, key: FieldKey) -> SymbolSpec:
        """Return the symbol for ``key``: its override, else the catalog's, else reject."""
        definition = fact_definition(key)
        return self.overrides.get(
            key,
            SymbolSpec(plain=definition.plain_symbol, math=definition.mathtext_symbol),
        )


@dataclass(frozen=True, slots=True)
class RenderProfile:
    """Controls field visibility, precision, and preferred line length."""

    name: str
    include_context: bool
    max_length: int | None = None
    precision: Mapping[FieldKey, int] = field(default_factory=dict)


LEGEND_PROFILE = RenderProfile(name="legend", include_context=False, max_length=55)
LOG_PROFILE = RenderProfile(name="log", include_context=True)
