"""Read a dataset's pinned test fold for comparison-time position lookups.

Pure reads over data Phase A (`composition/assignments/_split_pinning.py`)
already computed and persisted — no writing, no dlkit changes.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices

from neuralls.composition.assignments._split_pinning import SPLIT_FILENAME
from neuralls.platform.storage.dataset_readers import load_matrix_sample_index
from neuralls.shared.types import ComparisonRhsSourceKind


def resolve_test_row_indices(dataset_dir: Path) -> tuple[int, ...]:
    """Return the dataset's pinned test-fold row indices, in their stored order.

    Raises:
        ValueError: If the dataset has no pinned split yet — nothing has
            trained on it, so there is no test fold to address.
    """
    split_path = dataset_dir / SPLIT_FILENAME
    if not split_path.exists():
        raise ValueError(
            f"Dataset '{dataset_dir}' has no pinned split ({SPLIT_FILENAME}) yet; "
            "nothing has trained on it, so there is no test fold to address."
        )
    return load_split_indices(split_path).test


def resolve_test_matrix_indices(dataset_dir: Path) -> tuple[int, ...]:
    """Return the unique matrix indices bound to the dataset's pinned test-fold rows, sorted."""
    test_rows = resolve_test_row_indices(dataset_dir)
    matrix_sample_index = load_matrix_sample_index(dataset_dir)
    return tuple(sorted({int(matrix_sample_index[row]) for row in test_rows}))


def resolve_matrix_index[T: int | None](
    dataset_dir: Path,
    *,
    matrix_index: T,
    test_position: int | None,
    rhs_source_kind: ComparisonRhsSourceKind,
) -> int | T:
    """Return the raw matrix index one comparison entry addresses.

    The single place that decides which matrix an entry solves on, shared by
    group discovery (which proves setup-sharing safety from it) and the
    comparison runner (which loads that matrix), so the two cannot drift.

    A `test_position` selects a test-fold *matrix* unless the RHS source is the
    dataset itself — then the dataset is single-matrix and `test_position` only
    selects which RHS row to use, so `matrix_index` is already correct.

    Raises:
        IndexError: If `test_position` is beyond the pinned test fold.
    """
    if test_position is None or rhs_source_kind is ComparisonRhsSourceKind.DATASET:
        return matrix_index
    test_matrix_indices = resolve_test_matrix_indices(dataset_dir)
    if test_position >= len(test_matrix_indices):
        raise IndexError(
            f"test_position={test_position} is out of range for dataset "
            f"'{dataset_dir}', which has {len(test_matrix_indices)} test-fold matrices."
        )
    return test_matrix_indices[test_position]
