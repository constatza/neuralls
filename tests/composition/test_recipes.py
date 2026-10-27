"""RHS recipes and system tags: the one place a system's report identity is defined."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.composition.comparison.recipes import rhs_recipe, system_tag
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.comparison import (
    ComparisonRhsSourceModel,
    DatasetRhsSourceModel,
    GaussianRhsSourceModel,
    RandomMask,
    RawLhsSourceModel,
    RawRhsSourceModel,
    SinusoidalRhsSourceModel,
)
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    SinusoidalRecipe,
    recipe_key,
)
from neuralls.shared.types import DatasetId, MatrixId, RowKind


def test_different_omegas_are_different_recipes() -> None:
    assert rhs_recipe(SinusoidalRhsSourceModel(omega=6.283185)) == SinusoidalRecipe(6.283185)
    assert rhs_recipe(SinusoidalRhsSourceModel(omega=3.14)) != rhs_recipe(
        SinusoidalRhsSourceModel(omega=6.28)
    )


def test_identical_parameters_share_a_recipe() -> None:
    assert rhs_recipe(SinusoidalRhsSourceModel(omega=2.0)) == rhs_recipe(
        SinusoidalRhsSourceModel(omega=2.0)
    )


def test_a_mask_makes_a_separate_recipe_for_the_same_source() -> None:
    masked = rhs_recipe(GaussianRhsSourceModel(mask=RandomMask(density=0.1)))

    assert masked == GaussianRecipe(0.0, 1.0, RandomlyMasked(0.1))
    assert masked != rhs_recipe(GaussianRhsSourceModel())
    assert masked != rhs_recipe(GaussianRhsSourceModel(mask=RandomMask(density=0.2)))
    assert rhs_recipe(DatasetRhsSourceModel(mask=RandomMask())) == DatasetRecipe(
        RandomlyMasked(0.1)
    )


def test_gaussian_parameters_enter_the_recipe() -> None:
    assert rhs_recipe(GaussianRhsSourceModel(std=2.0)) == GaussianRecipe(0.0, 2.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            {"path": "/x", "sample_index": 0},
            {"path": "/y", "sample_index": 0},
        ),
        (
            {"path": "/x", "sample_index": 0},
            {"path": "/x", "sample_index": 1},
        ),
        (
            {"path": "/x", "row_kind": RowKind.STANDARD},
            {"path": "/x", "row_kind": RowKind.CG_INTERNAL},
        ),
        (
            {"path": "/x"},
            {"path": "/x", "scale": 2.0},
        ),
    ],
)
@pytest.mark.parametrize("model", [RawLhsSourceModel, RawRhsSourceModel])
def test_raw_sources_differing_in_any_identity_field_are_different_recipes(
    model: type[RawLhsSourceModel | RawRhsSourceModel],
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    one = rhs_recipe(model(**first))
    two = rhs_recipe(model(**second))

    assert one != two
    assert recipe_key(one) != recipe_key(two)


def test_the_same_raw_source_twice_is_one_recipe(tmp_path: Path) -> None:
    path = tmp_path / "vector.npy"
    assert rhs_recipe(RawLhsSourceModel(path=path)) == rhs_recipe(RawLhsSourceModel(path=path))


def test_raw_lhs_and_raw_rhs_of_one_file_are_different_recipes() -> None:
    assert rhs_recipe(RawLhsSourceModel(path="/x")) != rhs_recipe(RawRhsSourceModel(path="/x"))


def _planned(rhs_source: ComparisonRhsSourceModel, ordinal: int = 1) -> PlannedSystem:
    return PlannedSystem(
        entry=ComparisonRegistryEntry(
            id="sweep-7-1", matrix_dataset="eval", rhs_source=rhs_source.model_dump()
        ),
        matrix=MatrixRef(matrix_id=MatrixId(7), row=14),
        rhs_row=None,
        rhs_ordinal=ordinal,
        provenance=SharedSingleMatrix(raw_matrix_index=14),
    )


def test_system_tag_binds_dataset_recipe_matrix_ordinal_and_entry_id() -> None:
    tag = system_tag(_planned(SinusoidalRhsSourceModel(omega=2.0)))

    assert tag.dataset == DatasetId("eval")
    assert tag.recipe == SinusoidalRecipe(2.0)
    assert (tag.matrix_id, tag.rhs_ordinal, tag.system_id) == (7, 1, "sweep-7-1")
