"""The matrix-set-and-RHS-type, matrix-set and all-matrix-sets report levels, and the labels and keys they induce."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.shared.rhs_recipe import RhsRecipe, parse_recipe_key, recipe_key
from neuralls.shared.types import DatasetId

SYSTEM_ID = "system_id"


class ReportLevelName(StrEnum):
    """What a report figure groups by, as persisted in the figure manifest."""

    MATRIX_SET_RHS_TYPE = "matrix_set_rhs_type"
    MATRIX_SET = "matrix_set"
    ALL_MATRIX_SETS = "all_matrix_sets"
    METHOD = "method"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class MatrixSetRhsTypeLevel:
    """Group by matrix set (the case's `matrix_dataset`) and RHS type: one homogeneous problem family per group."""


@dataclass(frozen=True, slots=True)
class MatrixSetLevel:
    """Group by matrix set, pooling that set's RHS types."""


@dataclass(frozen=True, slots=True)
class AllMatrixSetsLevel:
    """No keys: pool every system of every matrix set in the case."""


type ReportLevel = MatrixSetRhsTypeLevel | MatrixSetLevel | AllMatrixSetsLevel


def report_key(level: ReportLevel, tag: SystemTag) -> tuple[str, ...]:
    """The values of ``tag`` that select its report group at ``level``."""
    match level:
        case MatrixSetRhsTypeLevel():
            return (tag.dataset, recipe_key(tag.recipe))
        case MatrixSetLevel():
            return (tag.dataset,)
        case AllMatrixSetsLevel():
            return ()


def level_name(label: GroupLabel) -> ReportLevelName:
    """The persisted name of the level that produced ``label``."""
    match label:
        case MatrixSetRhsTypeLabel():
            return ReportLevelName.MATRIX_SET_RHS_TYPE
        case MatrixSetLabel():
            return ReportLevelName.MATRIX_SET
        case AllMatrixSetsLabel():
            return ReportLevelName.ALL_MATRIX_SETS
        case _ as unreachable:
            assert_never(unreachable)


def label_key(label: GroupLabel) -> tuple[str, ...]:
    """The `report_key` that selects ``label``'s group (mixtures are not part of the key)."""
    match label:
        case MatrixSetRhsTypeLabel(dataset=dataset, recipe=recipe):
            return (dataset, recipe_key(recipe))
        case MatrixSetLabel(dataset=dataset):
            return (dataset,)
        case AllMatrixSetsLabel():
            return ()


def label_for_tags(level: ReportLevel, tags: Sequence[SystemTag]) -> GroupLabel:
    """The label of the group ``tags`` make up (all must share one `report_key`).

    Used for a group nothing was measured in, which `report_level` never sees.
    """
    mixture = sorted_recipes({tag.recipe for tag in tags})
    first = tags[0]
    match level:
        case MatrixSetRhsTypeLevel():
            return MatrixSetRhsTypeLabel(dataset=first.dataset, recipe=first.recipe)
        case MatrixSetLevel():
            return MatrixSetLabel(dataset=first.dataset, mixture_of=mixture)
        case AllMatrixSetsLevel():
            return AllMatrixSetsLabel(mixture_of=mixture)


@dataclass(frozen=True, slots=True)
class SystemCounts:
    """How many distinct systems, and distinct matrices, a report group measured."""

    n_systems: int
    n_matrices: int


def system_counts(frame: pl.DataFrame, level: ReportLevel) -> dict[GroupLabel, SystemCounts]:
    """Distinct measured systems and matrices of each group (a matrix is per dataset)."""
    keys = level_columns(level)
    counts = {
        "n_systems": pl.col(SYSTEM_ID).n_unique(),
        "n_matrices": pl.struct(["dataset", "matrix_id"]).n_unique(),
    }
    counted = (
        frame.group_by(keys, maintain_order=True).agg(**counts) if keys else frame.select(**counts)
    )
    mixtures = mixtures_by_key(frame, keys)
    return {
        group_label(level, row, mixtures): SystemCounts(
            n_systems=row["n_systems"], n_matrices=row["n_matrices"]
        )
        for row in counted.iter_rows(named=True)
    }


def level_columns(level: ReportLevel) -> list[str]:
    match level:
        case MatrixSetRhsTypeLevel():
            return ["dataset", "recipe"]
        case MatrixSetLevel():
            return ["dataset"]
        case AllMatrixSetsLevel():
            return []


def sorted_recipes(recipes: Iterable[RhsRecipe]) -> tuple[RhsRecipe, ...]:
    """The distinct ``recipes`` in machine-key order."""
    return tuple(sorted(set(recipes), key=recipe_key))


def _recipes_of_keys(keys: Iterable[str]) -> tuple[RhsRecipe, ...]:
    return sorted_recipes(parse_recipe_key(key) for key in keys)


def mixtures_by_key(
    frame: pl.DataFrame, keys: list[str]
) -> dict[tuple[str, ...], tuple[RhsRecipe, ...]]:
    """Recipes pooled by each group of ``keys``, in machine-key order."""
    if not keys:
        return {(): _recipes_of_keys(frame["recipe"].unique())}
    pooled = frame.group_by(keys, maintain_order=True).agg(mixture=pl.col("recipe").unique().sort())
    return {
        tuple(row[key] for key in keys): _recipes_of_keys(row["mixture"])
        for row in pooled.iter_rows(named=True)
    }


def group_label(
    level: ReportLevel,
    row: Mapping[str, object],
    mixtures: Mapping[tuple[str, ...], tuple[RhsRecipe, ...]],
) -> GroupLabel:
    match level:
        case MatrixSetRhsTypeLevel():
            return MatrixSetRhsTypeLabel(
                dataset=DatasetId(str(row["dataset"])), recipe=parse_recipe_key(str(row["recipe"]))
            )
        case MatrixSetLevel():
            dataset = str(row["dataset"])
            return MatrixSetLabel(dataset=DatasetId(dataset), mixture_of=mixtures[(dataset,)])
        case AllMatrixSetsLevel():
            return AllMatrixSetsLabel(mixture_of=mixtures[()])
