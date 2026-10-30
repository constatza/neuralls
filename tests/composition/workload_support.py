"""Pure builders for method-major workloads shared by composition tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.composition.comparison.models import MatrixWorkload, MatrixWorkloadMember
from neuralls.domain.solver.models import SystemTag
from neuralls.shared.types import DatasetId, MatrixId, SystemId
from tests.recipe_support import GAUSSIAN

DATASET = DatasetId("ds")
"""The matrix dataset every built bundle belongs to."""
MATRIX_PATH = Path("/workspace/not-a-dataset")
"""A path no test reads: extra inputs are only loaded for neural methods."""
DIAGONAL = (2.0, 3.0)
"""Diagonal of the matrix of matrix 1; matrix ``k`` is ``k`` times it."""


def matrix_bundle(matrix_id: int, *, n_rhs: int = 1) -> MatrixWorkload:
    """Build one diagonal-matrix bundle.

    Args:
        matrix_id: Matrix identity; also scales the diagonal.
        n_rhs: Number of sibling right-hand sides on the matrix.

    Returns:
        A workload whose members are ordered by RHS ordinal.
    """
    members = tuple(
        MatrixWorkloadMember(
            tag=SystemTag(
                DATASET,
                GAUSSIAN,
                MatrixId(matrix_id),
                ordinal,
                SystemId(f"ds-{matrix_id}-{ordinal}"),
            ),
            rhs=np.array([1.0, float(ordinal + 1)]),
        )
        for ordinal in range(n_rhs)
    )
    return MatrixWorkload(
        dataset=DATASET,
        matrix_id=MatrixId(matrix_id),
        matrix=np.diag(np.array(DIAGONAL)) * matrix_id,
        matrix_path=MATRIX_PATH,
        matrix_index=matrix_id,
        members=members,
    )
