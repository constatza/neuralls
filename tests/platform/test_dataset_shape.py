"""A dataset's matrix shape and held-out fold come from its stored bindings and split."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.domain.dataset_shape import MultiMatrix, SingleMatrix
from neuralls.platform.storage.dataset_shape import classify_dataset, load_held_out_fold
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.types import MatrixId


def test_two_matrices_classify_as_multi_matrix_with_their_ids(sparse_id_dataset: Path) -> None:
    assert classify_dataset(sparse_id_dataset) == MultiMatrix(matrix_ids=(MatrixId(5), MatrixId(9)))


def test_held_out_fold_groups_test_rows_by_their_matrix(sparse_id_dataset: Path) -> None:
    # Rows 0-1 are matrix 5 and rows 2-3 are matrix 9; hold out one of each plus a spare row.
    save_split_indices(
        IndexSplit(train=(1,), validation=(), test=(0, 2, 3), predict=None),
        sparse_id_dataset / SPLIT_FILENAME,
    )

    fold = load_held_out_fold(sparse_id_dataset)

    assert fold.matrices == (MatrixId(5), MatrixId(9))
    assert fold.rows_by_matrix == {MatrixId(5): (0,), MatrixId(9): (2, 3)}
    assert fold.rows == (0, 2, 3)


def test_held_out_fold_without_a_pinned_split_asks_for_generation(
    sparse_id_dataset: Path,
) -> None:
    with pytest.raises(StageInputsStale):
        load_held_out_fold(sparse_id_dataset)


def test_one_matrix_classifies_as_single_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from neuralls.platform.storage import dataset_shape

    monkeypatch.setattr(dataset_shape, "load_matrix_sample_index", lambda _dir: np.array([7, 7, 7]))

    assert classify_dataset(tmp_path) == SingleMatrix(matrix_id=MatrixId(7))
