"""Group ids are lossless in the recipe, so two recipes never share a file or metric key."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.labels import (
    FineLabel,
    GlobalLabel,
    MediumLabel,
    render_group_id,
)
from neuralls.shared.rhs_recipe import (
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawSourceIdentity,
    RhsRecipe,
    SinusoidalRecipe,
    parse_recipe_key,
)
from neuralls.shared.types import DatasetId, GroupId

DS = DatasetId("ds")


@pytest.fixture
def recipes() -> tuple[RhsRecipe, ...]:
    """Recipes that differ in one parameter or only in the mask."""
    return (
        GaussianRecipe(1.0, 2.0),
        GaussianRecipe(12.0, 2.0),
        GaussianRecipe(1.0, 2.0, RandomlyMasked(0.1)),
        GaussianRecipe(1.0, 2.0, RandomlyMasked(0.12)),
        SinusoidalRecipe(2.0),
        SinusoidalRecipe(20.0),
        RawLhsRecipe(RawSourceIdentity(0, "aa")),
        RawLhsRecipe(RawSourceIdentity(0, "ab")),
    )


def test_distinct_recipes_get_distinct_group_ids(recipes: tuple[RhsRecipe, ...]) -> None:
    ids = [render_group_id(FineLabel(DS, recipe)) for recipe in recipes]
    assert len(set(ids)) == len(ids)


def test_a_fine_group_id_returns_its_recipe(recipes: tuple[RhsRecipe, ...]) -> None:
    for recipe in recipes:
        group_id = render_group_id(FineLabel(DS, recipe))
        prefix, dataset, key = group_id.split("/")
        assert (prefix, dataset) == ("fine", "ds")
        assert parse_recipe_key(key) == recipe


def test_mixture_levels_are_named_by_level_and_dataset_only() -> None:
    mixture = (GaussianRecipe(0.0, 1.0), SinusoidalRecipe(2.0))
    assert render_group_id(MediumLabel(DS, mixture)) == GroupId("medium/ds")
    assert render_group_id(GlobalLabel(mixture)) == GroupId("global")
