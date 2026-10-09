"""Matrix leakage provenance for comparison inputs.

Tracks only whether a comparison's matrix was verified held-out of training
via the dataset's pinned split — a separate concern from
``ComparisonRhsSourceKind``/``RowKind``, which already track where the RHS
came from and whether a row is a genuine Ax=b pair. Nothing here duplicates
those fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.types import MatrixId, TestPosition


@dataclass(frozen=True, slots=True)
class PinnedSplitMatrix:
    """Matrix resolved from a `test_position` within the dataset's pinned split.

    Verified held-out: `test_position` addresses only `range(len(test_tuple))`
    of the dataset's `split.json`, so this can never resolve to a training row.
    """

    test_position: TestPosition
    raw_matrix_index: MatrixId


@dataclass(frozen=True, slots=True)
class UnverifiedMatrix:
    """Matrix resolved from a raw `matrix_index`, including the implicit default.

    Honest label for "no leakage guarantee was requested here" — not an error
    state, just the absence of a `test_position` request.
    """

    raw_matrix_index: MatrixId


@dataclass(frozen=True, slots=True)
class CrossDatasetMatrix:
    """Matrix loaded from an explicitly different dataset than the home comparison dataset.

    Never touches the home dataset's `split.json` — there is no "test fold"
    concept for data that was never part of this training run.
    """

    dataset_id: str
    raw_matrix_index: MatrixId


MatrixProvenance = PinnedSplitMatrix | UnverifiedMatrix | CrossDatasetMatrix
"""Closed set of ways a comparison's matrix leakage status can be known."""
