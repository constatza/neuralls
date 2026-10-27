"""Distinct RHS recipes shared by tests that build system tags and report labels."""

from __future__ import annotations

from neuralls.shared.rhs_recipe import GaussianRecipe, RhsRecipe, SinusoidalRecipe

GAUSSIAN: RhsRecipe = GaussianRecipe(mean=0.0, std=1.0)
"""The default standard-normal recipe."""
SINE: RhsRecipe = SinusoidalRecipe(omega=6.28319)
"""A second recipe, so mixtures have more than one member."""
ALPHA: RhsRecipe = GaussianRecipe(mean=0.0, std=2.0)
BETA: RhsRecipe = GaussianRecipe(mean=0.0, std=3.0)
