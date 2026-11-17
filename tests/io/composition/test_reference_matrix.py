"""The energy-POD reference matrix read from a stored dataset: the sparse train-fold mean only.

`sparse_id_dataset` stores `diag(5, 1)` on rows 0-1 (matrix id 5) and `diag(9, 1)` on
rows 2-3 (matrix id 9), unscaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.types.split import IndexSplit
from scipy.sparse import csr_array

from neuralls.composition.preconditioners.reference_matrix import mean_train_matrix
from neuralls.platform.storage.fit_data import FsFitDataLoader
from neuralls.shared.errors import StageInputsStale, StageInputsStaleReason
from tests.fakes.split_store import InMemorySplitStore


def _split(train: tuple[int, ...], test: tuple[int, ...]) -> IndexSplit:
    return IndexSplit(train=train, validation=(), test=test, predict=None)


@pytest.mark.parametrize(
    ("train", "test", "expected_diagonal"),
    [
        ((0, 1, 2), (3,), [7.0, 1.0]),
        ((0, 1), (2, 3), [5.0, 1.0]),
    ],
    ids=["both-matrices-in-train", "held-out-matrix-excluded"],
)
def test_mean_covers_each_train_fold_matrix_once(
    sparse_id_dataset: Path,
    train: tuple[int, ...],
    test: tuple[int, ...],
    expected_diagonal: list[float],
) -> None:
    store = InMemorySplitStore({sparse_id_dataset: _split(train, test)})

    mean = mean_train_matrix(sparse_id_dataset, store, FsFitDataLoader())

    assert isinstance(mean, csr_array)
    np.testing.assert_array_equal(mean.toarray(), np.diag(expected_diagonal))


def test_broadcast_matrix_is_read_from_its_single_stored_copy(
    single_matrix_dataset: Path,
) -> None:
    """Train rows 2-3 bind the one matrix stored at row 0 of the broadcast layout."""
    store = InMemorySplitStore({single_matrix_dataset: _split((2, 3), (0, 1))})

    mean = mean_train_matrix(single_matrix_dataset, store, FsFitDataLoader())

    np.testing.assert_array_equal(mean.toarray(), np.diag([7.0, 1.0]))


def test_a_dataset_without_a_pinned_split_is_stale(sparse_id_dataset: Path) -> None:
    with pytest.raises(StageInputsStale) as caught:
        mean_train_matrix(sparse_id_dataset, InMemorySplitStore(), FsFitDataLoader())

    assert caught.value.reason is StageInputsStaleReason.SPLIT_MISSING
