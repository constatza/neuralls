"""Pydantic boundary config for scientific-description rendering.

Converts TOML-sourced notation overrides into the pure
``neuralls.shared.scientific_description.policy`` types. Complete defaults mean an
ordinary config needs no ``[scientific_labels]`` section at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from neuralls.shared.scientific_description.models import FieldKey
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget, SymbolSpec

__all__ = ["ScientificLabelsConfig", "SymbolOverride"]


class SymbolOverride(BaseModel):
    """One field key's plain and math-mode symbol spelling, from TOML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plain: str
    math: str


class ScientificLabelsConfig(BaseModel):
    r"""``[scientific_labels]`` TOML section controlling notation and rendering.

    Example::

        [scientific_labels]
        rendering = "latex"

        [scientific_labels.symbols.sample_count]
        plain = "N_train"
        math = "N_{\\mathrm{train}}"
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rendering: RenderTarget = RenderTarget.MATHTEXT
    symbols: dict[str, SymbolOverride] = Field(default_factory=dict)

    def notation_policy(self) -> NotationPolicy:
        """Convert this boundary config into the pure ``NotationPolicy``."""
        overrides = {
            FieldKey(name): SymbolSpec(plain=override.plain, math=override.math)
            for name, override in self.symbols.items()
        }
        return NotationPolicy.default(self.rendering).with_overrides(overrides)
