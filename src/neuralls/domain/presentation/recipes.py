"""Render an RHS recipe: the kind in words, its parameters in the policy's notation."""

from __future__ import annotations

from typing import assert_never

from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawRhsRecipe,
    RawSourceIdentity,
    RecipeMask,
    RhsRecipe,
    SinusoidalRecipe,
    Unmasked,
)
from neuralls.shared.scientific_description.catalog import (
    FREQUENCY,
    MASK_DENSITY,
    MEAN,
    STANDARD_DEVIATION,
    make_fact,
)
from neuralls.shared.scientific_description.models import ScientificFact
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderProfile
from neuralls.shared.scientific_description.render import render_facts

__all__ = ["render_recipe"]

_GAUSSIAN = "Gaussian"
_SINUSOIDAL = "Sinusoidal"
_RAW_LHS = "Stored solution vector"
_RAW_RHS = "Stored RHS vector"
_DATASET = "Dataset RHS"
_STANDARD_NORMAL = (0.0, 1.0)
_FREQUENCY_PRECISION = 6


def render_recipe(recipe: RhsRecipe, profile: RenderProfile, policy: NotationPolicy) -> str:
    """The text of ``recipe``; the standard normal and an unmasked source show no parameters."""
    match recipe:
        case GaussianRecipe(mean=mean, std=std, mask=mask):
            facts: list[ScientificFact] = (
                []
                if (mean, std) == _STANDARD_NORMAL
                else [make_fact(MEAN, mean), make_fact(STANDARD_DEVIATION, std)]
            )
            return _compose(_GAUSSIAN, facts, (), mask, profile, policy)
        case SinusoidalRecipe(omega=omega, mask=mask):
            return _compose(_SINUSOIDAL, [make_fact(FREQUENCY, omega)], (), mask, profile, policy)
        case RawLhsRecipe(source=source, mask=mask):
            return _compose(_RAW_LHS, [], (_sample(source),), mask, profile, policy)
        case RawRhsRecipe(source=source, mask=mask):
            return _compose(_RAW_RHS, [], (_sample(source),), mask, profile, policy)
        case DatasetRecipe(mask=mask):
            return _compose(_DATASET, [], (), mask, profile, policy)
        case _:
            assert_never(recipe)


def _sample(source: RawSourceIdentity) -> str:
    return f"sample {source.sample_index}"


def _compose(
    name: str,
    facts: list[ScientificFact],
    words: tuple[str, ...],
    mask: RecipeMask,
    profile: RenderProfile,
    policy: NotationPolicy,
) -> str:
    match mask:
        case Unmasked():
            pass
        case RandomlyMasked(density=density):
            facts.append(make_fact(MASK_DENSITY, density))
        case _:
            assert_never(mask)
    precision = {FREQUENCY: _FREQUENCY_PRECISION, **profile.precision}
    rendered = render_facts(tuple(facts), policy, precision=precision)
    details = ", ".join(part for part in (*words, rendered) if part)
    return f"{name} ({details})" if details else name
