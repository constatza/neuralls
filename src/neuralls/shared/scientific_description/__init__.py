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
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BOOTSTRAP_CYCLES,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COARSE_DIMENSION,
    COARSENING_STRATEGY,
    DAMPING,
    DEFAULT_SYMBOLS,
    DETAILED_PROFILE,
    DROP_TOLERANCE,
    LEGEND_PROFILE,
    LOG_PROFILE,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SMOOTHER_DAMPING,
    THRESHOLD,
    TITLE_PROFILE,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
    SymbolSpec,
)
from neuralls.shared.scientific_description.render import render, render_facts

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
    "LEGEND_PROFILE",
    "LOG_PROFILE",
    "MODEL_ARCHITECTURE",
    "MULTIGRID_LEVELS",
    "POST_SMOOTHING_STEPS",
    "PRE_SMOOTHING_STEPS",
    "RELAXATION_SWEEPS",
    "RELAXATION_TEST_VECTORS",
    "SMOOTHER_DAMPING",
    "THRESHOLD",
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
