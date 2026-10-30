"""Every report scope sits in one containment chain from a system up to all matrix sets."""

from __future__ import annotations

from neuralls.domain.solver.models.labels import MatrixSetLabel, MatrixSetRhsTypeLabel
from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    System,
    parent_scope,
    scope_of_label,
)
from neuralls.shared.types import DatasetId, MatrixId, SystemId
from tests.recipe_support import GAUSSIAN

DATASET = DatasetId("ds")


def test_a_system_is_contained_by_its_cohort_set_and_the_root() -> None:
    scope = System(DATASET, GAUSSIAN, MatrixId(3), 7, SystemId("s"))

    chain = [scope]
    while (parent := parent_scope(chain[-1])) is not None:
        chain.append(parent)

    assert chain[1:] == [MatrixSetRhsType(DATASET, GAUSSIAN), MatrixSet(DATASET), AllMatrixSets()]


def test_group_labels_map_to_their_scopes() -> None:
    assert scope_of_label(MatrixSetRhsTypeLabel(DATASET, GAUSSIAN)) == MatrixSetRhsType(
        DATASET, GAUSSIAN
    )
    assert scope_of_label(MatrixSetLabel(DATASET, (GAUSSIAN,))) == MatrixSet(DATASET)
