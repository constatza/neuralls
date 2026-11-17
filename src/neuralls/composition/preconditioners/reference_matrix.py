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
from neuralls.composition.generation.split_pinning import pinned_train_rows
from neuralls.composition.preconditioners.ports import FitDataLoader


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


def mean_train_matrix(
    dataset_dir: Path, split_store: SplitStore, fit_data: FitDataLoader
) -> csr_array:
    """Sparse mean of the distinct scaled matrices bound to the dataset's train fold.

    The stored matrices are already divided by their scale factors, so this is ``Ā``
    in the scaled space the snapshots were generated in. Matrices with different
    sparsity patterns are summed over the union pattern. It stays CSR, because the
    energy fit only multiplies by it.

    Args:
        dataset_dir: Generated dataset directory with a pinned split.
        split_store: Where the dataset's split is pinned.
        fit_data: Where the dataset's matrices are read from.

    Returns:
        The ``(n, n)`` float64 mean matrix, in CSR.

    Raises:
        StageInputsStale: If the dataset has no pinned split.
        ValueError: If the train fold is empty.
    """
    rows = train_matrix_rows(
        fit_data.matrix_sample_index(dataset_dir), pinned_train_rows(dataset_dir, split_store)
    )
    # Running sum: one matrix in memory at a time, however many the train fold binds.
    bindings = iter(rows.items())
    matrix_id, row = next(bindings)
    total = csr_array(fit_data.scaled_matrix(dataset_dir, row, matrix_id), dtype=np.float64)
    for matrix_id, row in bindings:
        total = total + fit_data.scaled_matrix(dataset_dir, row, matrix_id)
    return csr_array(total / len(rows))
