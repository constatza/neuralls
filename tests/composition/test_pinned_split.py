"""Reading a dataset's pinned test fold for `test_position` translation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.composition.comparison import pinned_split
from neuralls.composition.comparison.pinned_split import (
    resolve_test_matrix_indices,
    resolve_test_row_indices,
)


def _write_split(dataset_dir: Path, split: IndexSplit) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(split, dataset_dir / "split.json")


def test_resolve_test_row_indices_reads_split_test_fold(tmp_path: Path) -> None:
    split = IndexSplit(train=(0, 1, 2), validation=(3,), test=(4, 5), predict=None)
    _write_split(tmp_path, split)

    assert resolve_test_row_indices(tmp_path) == (4, 5)


def test_resolve_test_row_indices_without_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing has trained"):
        resolve_test_row_indices(tmp_path)


def test_resolve_test_matrix_indices_derives_unique_sorted_matrix_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = IndexSplit(train=(0, 1, 2), validation=(), test=(3, 4, 5), predict=None)
    _write_split(tmp_path, split)
    matrix_sample_index = np.array([0, 0, 0, 2, 1, 1], dtype=np.int64)
    monkeypatch.setattr(
        pinned_split, "load_matrix_sample_index", lambda _dataset_dir: matrix_sample_index
    )

    assert resolve_test_matrix_indices(tmp_path) == (1, 2)


def test_resolve_test_matrix_indices_without_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing has trained"):
        resolve_test_matrix_indices(tmp_path)
