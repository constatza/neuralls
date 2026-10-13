"""Standard-library-only scientific description and rendering core.

See ``docs/plan.md`` for the full vocabulary redesign this package is the
pure foundation of.
"""

from __future__ import annotations

from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
)
from neuralls.shared.scientific_description.policy import (
    DEFAULT_SYMBOLS,
    DETAILED_PROFILE,
    LEGEND_PROFILE,
    LOG_PROFILE,
    TITLE_PROFILE,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
    SymbolSpec,
)
from neuralls.shared.scientific_description.render import render, render_facts

__all__ = [
    "DEFAULT_SYMBOLS",
    "DETAILED_PROFILE",
    "LEGEND_PROFILE",
    "LOG_PROFILE",
    "TITLE_PROFILE",
    "DecimalFact",
    "FieldImportance",
    "FieldKey",
    "IntegerFact",
    "NotationPolicy",
    "RenderProfile",
    "RenderTarget",
    "ScientificDescription",
    "ScientificFact",
    "SymbolSpec",
    "TextFact",
    "render",
    "render_facts",
]
