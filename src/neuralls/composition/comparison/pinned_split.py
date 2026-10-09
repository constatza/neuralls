"""Read a dataset's pinned test fold for comparison-time position lookups.

Pure reads over data Phase A (`composition/assignments/_split_pinning.py`)
already computed and persisted — no writing, no dlkit changes.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices

from neuralls.composition.assignments._split_pinning import SPLIT_FILENAME
from neuralls.platform.storage.dataset_readers import load_matrix_sample_index


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
