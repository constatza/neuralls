"""The energy-POD reference matrix read from a stored dataset: the train-fold mean only.

`sparse_id_dataset` stores `diag(5, 1)` on rows 0-1 (matrix id 5) and `diag(9, 1)` on
rows 2-3 (matrix id 9), unscaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.preconditioners.pod_fittable import PODCoarseningFittable
from neuralls.composition.preconditioners.reference_matrix import mean_train_matrix
from neuralls.shared.constants import SPLIT_FILENAME
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

    mean = mean_train_matrix(sparse_id_dataset, store)

    np.testing.assert_array_equal(mean, np.diag(expected_diagonal))


def test_broadcast_matrix_is_read_from_its_single_stored_copy(
    single_matrix_dataset: Path,
) -> None:
    """Train rows 2-3 bind the one matrix stored at row 0 of the broadcast layout."""
    store = InMemorySplitStore({single_matrix_dataset: _split((2, 3), (0, 1))})

    mean = mean_train_matrix(single_matrix_dataset, store)

    np.testing.assert_array_equal(mean, np.diag([7.0, 1.0]))


def test_a_dataset_without_a_pinned_split_is_stale(sparse_id_dataset: Path) -> None:
    with pytest.raises(StageInputsStale) as caught:
        mean_train_matrix(sparse_id_dataset, InMemorySplitStore())

    assert caught.value.reason is StageInputsStaleReason.SPLIT_MISSING


def test_energy_fit_weighs_snapshots_with_the_pinned_train_mean(
    sparse_id_dataset: Path,
) -> None:
    save_split_indices(_split((0, 1, 2), (3,)), sparse_id_dataset / SPLIT_FILENAME)
    snapshots = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64)
    reference = PODCoarseningStrategy(rank=1)
    reference.fit(
        snapshots, inner_product=torch.diag(torch.tensor([7.0, 1.0], dtype=torch.float64))
    )
    model = PODCoarseningFittable(
        rank=1, inner_product={"kind": "energy", "dataset_dir": str(sparse_id_dataset)}
    )

    model.fit(snapshots)

    torch.testing.assert_close(model._basis, reference._basis, rtol=0.0, atol=0.0)
