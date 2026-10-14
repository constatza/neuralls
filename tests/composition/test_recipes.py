"""RHS recipe labels and system tags: the one place a system's report identity is defined."""

from __future__ import annotations

import pytest

from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.composition.comparison.recipes import system_tag
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.comparison import (
    DatasetRhsSourceModel,
    GaussianRhsSourceModel,
    RandomMask,
    RawLhsSourceModel,
    RawRhsSourceModel,
    SinusoidalRhsSourceModel,
)
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.shared.types import DatasetId, MatrixId


def test_different_omegas_are_different_recipes() -> None:
    assert SinusoidalRhsSourceModel(omega=6.283185).recipe_label == "sinusoidal ω=6.28318"
    assert (
        SinusoidalRhsSourceModel(omega=3.14).recipe_label
        != SinusoidalRhsSourceModel(omega=6.28).recipe_label
    )


def test_identical_parameters_share_a_recipe() -> None:
    assert (
        SinusoidalRhsSourceModel(omega=2.0).recipe_label
        == SinusoidalRhsSourceModel(omega=2.0).recipe_label
    )


def test_a_mask_makes_a_separate_recipe_for_the_same_source() -> None:
    plain = GaussianRhsSourceModel()
    masked = GaussianRhsSourceModel(mask=RandomMask(density=0.1))

    assert masked.recipe_label == "gaussian · mask ρ=0.1"
    assert plain.recipe_label == "gaussian"
    assert masked.recipe_label != GaussianRhsSourceModel(mask=RandomMask(density=0.2)).recipe_label
    assert DatasetRhsSourceModel(mask=RandomMask()).recipe_label == "dataset · mask ρ=0.1"


def test_default_gaussian_is_just_its_kind_and_a_non_default_one_shows_parameters() -> None:
    assert GaussianRhsSourceModel().recipe_label == "gaussian"
    assert GaussianRhsSourceModel(std=2.0).recipe_label == "gaussian[μ=0, σ=2]"


@pytest.mark.parametrize(
    ("source", "label"),
    [
        (RawLhsSourceModel(path="/x"), "raw_lhs"),
        (RawRhsSourceModel(path="/x"), "raw_rhs"),
        (DatasetRhsSourceModel(), "dataset"),
    ],
)
def test_fixed_and_dataset_recipes_are_their_kind(
    source: RawLhsSourceModel | RawRhsSourceModel | DatasetRhsSourceModel,
    label: str,
) -> None:
    assert source.recipe_label == label


def test_system_tag_binds_dataset_recipe_matrix_ordinal_and_entry_id() -> None:
    planned = PlannedSystem(
        entry=ComparisonRegistryEntry(
            id="sweep-7-1",
            matrix_dataset="eval",
            rhs_source={"kind": "sinusoidal", "omega": 2.0},
        ),
        matrix=MatrixRef(matrix_id=MatrixId(7), row=14),
        rhs_row=None,
        rhs_ordinal=1,
        provenance=SharedSingleMatrix(raw_matrix_index=14),
    )

    tag = system_tag(planned)

    assert tag.dataset == DatasetId("eval")
    assert tag.recipe == "sinusoidal ω=2"
    assert (tag.matrix_id, tag.rhs_ordinal, tag.system_id) == (7, 1, "sweep-7-1")
