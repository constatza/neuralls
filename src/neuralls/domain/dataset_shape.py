"""How a dataset's rows relate to its matrices, and which of them are held out.

Pure value types; the readers that build them from a dataset directory live in
`platform/storage/dataset_shape.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.shared.types import MatrixId


@dataclass(frozen=True, slots=True)
class SingleMatrix:
    """Every row is a right-hand side for the same one matrix."""

    matrix_id: MatrixId


@dataclass(frozen=True, slots=True)
class MultiMatrix:
    """Rows are bound to several distinct matrices (one matrix, many rows)."""

    matrix_ids: tuple[MatrixId, ...]


type DatasetShape = SingleMatrix | MultiMatrix


@dataclass(frozen=True, slots=True)
class HeldOutFold:
    """The pinned test fold, grouped by the matrix each test row is bound to.

    `rows_by_matrix` maps every held-out matrix to its test rows, in stored
    order, so a matrix with several `(lhs, rhs)` rows keeps all of them.
    """

    rows_by_matrix: Mapping[MatrixId, tuple[int, ...]]

    @property
    def matrices(self) -> tuple[MatrixId, ...]:
        """Held-out matrix ids in ascending order."""
        return tuple(sorted(self.rows_by_matrix))

    @property
    def rows(self) -> tuple[int, ...]:
        """Every held-out row, matrix by matrix in ascending matrix-id order."""
        return tuple(row for matrix in self.matrices for row in self.rows_by_matrix[matrix])
