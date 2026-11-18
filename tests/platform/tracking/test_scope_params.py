"""Scope params name the scope of a run and nothing that varies between its figures."""

from __future__ import annotations

import pytest
from tests.recipe_support import GAUSSIAN

from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    ReportScope,
)
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.types import DatasetId

DATASET = DatasetId("ds")
RECIPE_KEY = str(recipe_key(GAUSSIAN))


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (AllMatrixSets(), {"report_level": "all_matrix_sets"}),
        (MatrixSet(DATASET), {"report_level": "matrix_set", "dataset_id": "ds"}),
        (
            MatrixSetRhsType(DATASET, GAUSSIAN),
            {"report_level": "matrix_set_rhs_type", "dataset_id": "ds", "recipe_key": RECIPE_KEY},
        ),
    ],
)
def test_scope_params_hold_the_identity_of_the_scope(
    scope: ReportScope, expected: dict[str, str]
) -> None:
    assert scope_params(scope) == expected
