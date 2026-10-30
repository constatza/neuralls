"""Group ids are lossless in the recipe, so two recipes never share a file or metric key."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.reporting.report_levels import ReportLevelName, level_name
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
    ids = [render_group_id(MatrixSetRhsTypeLabel(DS, recipe)) for recipe in recipes]
    assert len(set(ids)) == len(ids)


def test_a_matrix_set_rhs_type_group_id_returns_its_recipe(recipes: tuple[RhsRecipe, ...]) -> None:
    for recipe in recipes:
        group_id = render_group_id(MatrixSetRhsTypeLabel(DS, recipe))
        prefix, dataset, key = group_id.split("/")
        assert (prefix, dataset) == ("matrix_set_rhs", "ds")
        assert parse_recipe_key(key) == recipe


def test_mixture_levels_are_named_by_level_and_dataset_only() -> None:
    mixture = (GaussianRecipe(0.0, 1.0), SinusoidalRecipe(2.0))
    assert render_group_id(MatrixSetLabel(DS, mixture)) == GroupId("matrix_set/ds")
    assert render_group_id(AllMatrixSetsLabel(mixture)) == GroupId("all")


def test_level_name_covers_every_label_variant(recipes: tuple[RhsRecipe, ...]) -> None:
    assert level_name(MatrixSetRhsTypeLabel(DS, recipes[0])) is ReportLevelName.MATRIX_SET_RHS_TYPE
    assert level_name(MatrixSetLabel(DS, recipes)) is ReportLevelName.MATRIX_SET
    assert level_name(AllMatrixSetsLabel(recipes)) is ReportLevelName.ALL_MATRIX_SETS
