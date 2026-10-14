"""Where a comparison's matrix came from, and what that says about leakage.

Tracks only the matrix side — ``ComparisonRhsSourceKind``/``RowKind`` already
track where the RHS came from and whether a row is a genuine Ax=b pair.
Every variant is a claim the planner can actually back; there is no
"unverified" case.
"""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.types import MatrixId


@dataclass(frozen=True, slots=True)
class SharedSingleMatrix:
    """The dataset's one and only matrix.

    It is in both the training and the test fold by construction, so no
    held-out claim is made; what the pinned split holds out is only the RHS rows.
    """

    raw_matrix_index: int


@dataclass(frozen=True, slots=True)
class HeldOutMatrix:
    """A matrix of a multi-matrix dataset whose rows all sit in the pinned test fold."""

    matrix_id: MatrixId
    raw_matrix_index: int


MatrixProvenance = SharedSingleMatrix | HeldOutMatrix
"""Closed set of ways a comparison's matrix leakage status is known."""
