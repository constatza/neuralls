"""The reference matrix of an energy-norm POD fit: the mean of a dataset's train-fold matrices.

A pooled snapshot family has no single energy norm, so the energy POD weighs every snapshot
with ``Ā = mean_i A_i'``, the mean of the distinct scaled matrices the train fold binds. Only
train-fold matrices count, so a held-out matrix never shapes the basis it is tested with.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from neuralls.composition.generation.ports import SplitStore
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sample_index,
    load_matrix_sparse_sample,
    matrix_row,
    resolve_dataset_artifacts,
)
from neuralls.shared.errors import StageInputsStale, StageInputsStaleReason


def train_matrix_rows(matrix_sample_index: np.ndarray, train_rows: Sequence[int]) -> dict[int, int]:
    """The first train row of each distinct matrix id the train fold binds, by matrix id.

    Args:
        matrix_sample_index: Matrix id bound to every dataset row.
        train_rows: Dataset rows of the train fold.

    Returns:
        Matrix id to the first train row bound to it, in ascending id order.
    """
    first_rows: dict[int, int] = {}
    for row in sorted(train_rows):
        first_rows.setdefault(int(matrix_sample_index[row]), row)
    return dict(sorted(first_rows.items()))


def mean_train_matrix(dataset_dir: Path, split_store: SplitStore) -> np.ndarray:
    """Dense mean of the distinct scaled matrices bound to the dataset's train fold.

    The stored matrices are already divided by their scale factors, so this is ``Ā``
    in the scaled space the snapshots were generated in. Matrices with different
    sparsity patterns are summed over the union pattern.

    Args:
        dataset_dir: Generated dataset directory with a pinned split.
        split_store: Where the dataset's split is pinned.

    Returns:
        The ``(n, n)`` mean matrix.

    Raises:
        StageInputsStale: If the dataset has no pinned split.
        ValueError: If the train fold is empty.
    """
    split = split_store.read(dataset_dir)
    if split is None:
        raise StageInputsStale(
            StageInputsStaleReason.SPLIT_MISSING, f"Dataset '{dataset_dir}' has no pinned split"
        )
    rows = train_matrix_rows(load_matrix_sample_index(dataset_dir), split.train)
    if not rows:
        raise ValueError(f"Dataset '{dataset_dir}' has an empty train fold.")
    artifact = resolve_dataset_artifacts(dataset_dir).matrix

    def load(matrix_id: int, row: int) -> csr_array:
        physical = matrix_row(artifact, row_index=row, matrix_id=matrix_id)
        return load_matrix_sparse_sample(dataset_dir, physical)

    # Running sum: one matrix in memory at a time, however many the train fold binds.
    bindings = iter(rows.items())
    total = load(*next(bindings))
    for matrix_id, row in bindings:
        total = total + load(matrix_id, row)
    return np.asarray(total.toarray(), dtype=np.float64) / len(rows)
