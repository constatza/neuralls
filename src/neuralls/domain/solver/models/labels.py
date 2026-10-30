"""Report-group labels and their stable string ids."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.types import DatasetId, GroupId


@dataclass(frozen=True, slots=True)
class MatrixSetRhsTypeLabel:
    """One matrix set and one RHS type: a homogeneous problem family."""

    dataset: DatasetId
    recipe: RhsRecipe


@dataclass(frozen=True, slots=True)
class MatrixSetLabel:
    """One matrix set, pooling every RHS type listed in `mixture_of`."""

    dataset: DatasetId
    mixture_of: tuple[RhsRecipe, ...]


@dataclass(frozen=True, slots=True)
class AllMatrixSetsLabel:
    """Every matrix set of the case, pooling the RHS types in `mixture_of`."""

    mixture_of: tuple[RhsRecipe, ...]


type GroupLabel = MatrixSetRhsTypeLabel | MatrixSetLabel | AllMatrixSetsLabel
"""Identifies the keys of one report group. Matrix-set and all-matrix-sets
groups are mixtures: their spread mostly reflects RHS types differing, so titles must say so."""


def render_group_id(label: GroupLabel) -> GroupId:
    """The one stable, level-unique string form of a label (filename stem, metric key segment).

    The level prefix keeps a matrix set named like another level's id from
    colliding with it, and the recipe's machine key is lossless, so two
    recipes never share an id.
    """
    match label:
        case MatrixSetRhsTypeLabel(dataset=dataset, recipe=recipe):
            return GroupId(f"matrix_set_rhs/{dataset}/{recipe_key(recipe)}")
        case MatrixSetLabel(dataset=dataset):
            return GroupId(f"matrix_set/{dataset}")
        case AllMatrixSetsLabel():
            return GroupId("all")
