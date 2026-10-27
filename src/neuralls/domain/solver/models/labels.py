"""Report-group labels and their stable string ids."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.types import DatasetId, GroupId


@dataclass(frozen=True, slots=True)
class FineLabel:
    """One dataset and one RHS recipe: a homogeneous problem family."""

    dataset: DatasetId
    recipe: RhsRecipe


@dataclass(frozen=True, slots=True)
class MediumLabel:
    """One dataset, pooling every RHS recipe listed in `mixture_of`."""

    dataset: DatasetId
    mixture_of: tuple[RhsRecipe, ...]


@dataclass(frozen=True, slots=True)
class GlobalLabel:
    """Every dataset and recipe of the case, pooling the recipes in `mixture_of`."""

    mixture_of: tuple[RhsRecipe, ...]


type GroupLabel = FineLabel | MediumLabel | GlobalLabel
"""Identifies the keys of one report group. Medium and global groups are
mixtures: their spread mostly reflects recipes differing, so titles must say so."""


def render_group_id(label: GroupLabel) -> GroupId:
    """The one stable, level-unique string form of a label (filename stem, metric key segment).

    The level prefix keeps a dataset named like another level's id from
    colliding with it, and the recipe's machine key is lossless, so two
    recipes never share an id.
    """
    match label:
        case FineLabel(dataset=dataset, recipe=recipe):
            return GroupId(f"fine/{dataset}/{recipe_key(recipe)}")
        case MediumLabel(dataset=dataset):
            return GroupId(f"medium/{dataset}")
        case GlobalLabel():
            return GroupId("global")
