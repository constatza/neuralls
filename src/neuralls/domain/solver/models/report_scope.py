"""Where a report result sits in the aggregation hierarchy, most aggregated at the top."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.types import DatasetId, MatrixId, SystemId


@dataclass(frozen=True, slots=True)
class AllMatrixSets:
    """Every matrix set of the session: the root of a report."""


@dataclass(frozen=True, slots=True)
class MatrixSet:
    """One matrix set, pooling all its RHS types."""

    dataset: DatasetId


@dataclass(frozen=True, slots=True)
class MatrixSetRhsType:
    """One matrix set and one RHS type: the primary analysis level."""

    dataset: DatasetId
    recipe: RhsRecipe


@dataclass(frozen=True, slots=True)
class System:
    """One (matrix, RHS) system, the least aggregated level."""

    dataset: DatasetId
    recipe: RhsRecipe
    matrix_id: MatrixId
    rhs_ordinal: int
    system_id: SystemId


type ReportScope = AllMatrixSets | MatrixSet | MatrixSetRhsType | System
"""The closed set of levels a report is organised by; each has one containing parent."""


def parent_scope(scope: ReportScope) -> ReportScope | None:
    """The scope containing ``scope``, or ``None`` for the root."""
    match scope:
        case AllMatrixSets():
            return None
        case MatrixSet():
            return AllMatrixSets()
        case MatrixSetRhsType(dataset=dataset):
            return MatrixSet(dataset)
        case System(dataset=dataset, recipe=recipe):
            return MatrixSetRhsType(dataset, recipe)
        case unreachable:
            assert_never(unreachable)


def scope_of_label(label: GroupLabel) -> ReportScope:
    """The report scope a group label aggregates over."""
    match label:
        case MatrixSetRhsTypeLabel(dataset=dataset, recipe=recipe):
            return MatrixSetRhsType(dataset, recipe)
        case MatrixSetLabel(dataset=dataset):
            return MatrixSet(dataset)
        case AllMatrixSetsLabel():
            return AllMatrixSets()
        case unreachable:
            assert_never(unreachable)
