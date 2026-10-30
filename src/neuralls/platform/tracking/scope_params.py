"""MLflow params naming the scope a report run covers.

Only facts constant over every figure of the run are params; anything that can differ
between figures of one run (kind, scenario, solver settings) lives in the figure cards.
"""

from __future__ import annotations

from typing import assert_never

from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    ReportScope,
    System,
)
from neuralls.domain.solver.reporting.report_levels import level_of_scope
from neuralls.shared.rhs_recipe import recipe_key

PARAM_REPORT_LEVEL = "report_level"
PARAM_DATASET_ID = "dataset_id"
PARAM_RECIPE_KEY = "recipe_key"
PARAM_MATRIX_ID = "matrix_id"
PARAM_RHS_ORDINAL = "rhs_ordinal"


def scope_params(scope: ReportScope) -> dict[str, str]:
    """The scope identity of a report run, as MLflow param values."""
    level = {PARAM_REPORT_LEVEL: level_of_scope(scope).value}
    match scope:
        case AllMatrixSets():
            return level
        case MatrixSet(dataset=dataset):
            return {**level, PARAM_DATASET_ID: str(dataset)}
        case MatrixSetRhsType(dataset=dataset, recipe=recipe):
            return {
                **level,
                PARAM_DATASET_ID: str(dataset),
                PARAM_RECIPE_KEY: str(recipe_key(recipe)),
            }
        case System(dataset=dataset, recipe=recipe, matrix_id=matrix_id, rhs_ordinal=ordinal):
            return {
                **level,
                PARAM_DATASET_ID: str(dataset),
                PARAM_RECIPE_KEY: str(recipe_key(recipe)),
                PARAM_MATRIX_ID: str(int(matrix_id)),
                PARAM_RHS_ORDINAL: str(ordinal),
            }
        case unreachable:
            assert_never(unreachable)
