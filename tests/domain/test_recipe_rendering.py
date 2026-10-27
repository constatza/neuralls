"""RHS recipes render through the notation policy, never as enum values."""

from __future__ import annotations

import pytest

from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.recipes import render_recipe
from neuralls.domain.solver.models.figure_scope import RecipeScope, render_figure_scope
from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawRhsRecipe,
    RawSourceIdentity,
    RhsRecipe,
    SinusoidalRecipe,
)
from neuralls.shared.scientific_description.catalog import STANDARD_DEVIATION
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    NotationPolicy,
    RenderTarget,
    SymbolSpec,
)

RAW = RawSourceIdentity(sample_index=3, fingerprint="0123456789abcdef")


@pytest.mark.parametrize(
    ("recipe", "plain", "mathtext"),
    [
        (GaussianRecipe(0.0, 1.0), "Gaussian", "Gaussian"),
        (
            GaussianRecipe(0.5, 2.0),
            "Gaussian (μ=0.5, σ=2)",
            "Gaussian ($\\mu=0.5$, $\\sigma=2$)",
        ),
        (
            SinusoidalRecipe(6.283185),
            "Sinusoidal (ω=6.28318)",
            "Sinusoidal ($\\omega=6.28318$)",
        ),
        (
            GaussianRecipe(0.0, 1.0, mask=RandomlyMasked(0.1)),
            "Gaussian (ρ=0.1)",
            "Gaussian ($\\rho=0.1$)",
        ),
        (
            RawLhsRecipe(RAW),
            "Stored solution vector (sample 3)",
            "Stored solution vector (sample 3)",
        ),
        (RawRhsRecipe(RAW), "Stored RHS vector (sample 3)", "Stored RHS vector (sample 3)"),
        (DatasetRecipe(), "Dataset RHS", "Dataset RHS"),
        (
            DatasetRecipe(mask=RandomlyMasked(0.25)),
            "Dataset RHS (ρ=0.25)",
            "Dataset RHS ($\\rho=0.25$)",
        ),
    ],
)
def test_recipe_renders_from_the_notation_policy(
    recipe: RhsRecipe, plain: str, mathtext: str
) -> None:
    assert render_recipe(recipe, LEGEND_PROFILE, NotationPolicy.default()) == plain
    assert (
        render_recipe(recipe, LEGEND_PROFILE, NotationPolicy.default(RenderTarget.MATHTEXT))
        == mathtext
    )


def test_a_notation_override_changes_the_recipe_glyph() -> None:
    policy = NotationPolicy.default().with_overrides({STANDARD_DEVIATION: SymbolSpec("sd", "sd")})
    assert render_recipe(GaussianRecipe(0.0, 2.0), LEGEND_PROFILE, policy) == "Gaussian (μ=0, sd=2)"


def test_recipe_scopes_never_print_a_kind_identifier(
    dataset_presentation: DatasetPresentation,
) -> None:
    texts = [
        render_figure_scope(
            RecipeScope(dataset_presentation, recipe), LEGEND_PROFILE, NotationPolicy.default()
        ).text
        for recipe in (RawLhsRecipe(RAW), RawRhsRecipe(RAW), DatasetRecipe())
    ]

    assert [text.rpartition(" - ")[2] for text in texts] == [
        "Stored solution vector (sample 3)",
        "Stored RHS vector (sample 3)",
        "Dataset RHS",
    ]
