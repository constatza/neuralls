"""Pinned-split computation is deterministic, idempotent, and matrix-group-aware."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.types.split import RatioSplitStrategy

from neuralls.composition.assignments import _split_pinning
from neuralls.composition.assignments._split_pinning import resolve_or_create_pinned_split

_SEED = 20240521
_TEST_RATIO = 0.2
_VAL_RATIO = 0.2


@pytest.fixture
def single_matrix_index() -> np.ndarray:
    return np.zeros(10, dtype=np.int64)


@pytest.fixture
def many_matrix_index() -> np.ndarray:
    # Matrix 0 has 3 rows, matrix 1 has 2 rows, matrix 2 has 1 row.
    return np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)


def _patch_matrix_sample_index(monkeypatch: pytest.MonkeyPatch, values: np.ndarray) -> None:
    monkeypatch.setattr(_split_pinning, "load_matrix_sample_index", lambda _dataset_dir: values)


def test_single_matrix_delegates_to_ratio_split_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_matrix_index: np.ndarray,
) -> None:
    _patch_matrix_sample_index(monkeypatch, single_matrix_index)

    split_path = resolve_or_create_pinned_split(
        tmp_path, test_ratio=_TEST_RATIO, val_ratio=_VAL_RATIO, seed=_SEED
    )

    expected = RatioSplitStrategy(
        num_samples=len(single_matrix_index),
        test_ratio=_TEST_RATIO,
        val_ratio=_VAL_RATIO,
        seed=_SEED,
    ).split()
    actual = json.loads(split_path.read_text())
    assert sorted(actual["train"]) == sorted(expected.train)
    assert sorted(actual["validation"]) == sorted(expected.validation)
    assert sorted(actual["test"]) == sorted(expected.test)


def test_many_matrix_keeps_each_matrixs_rows_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    many_matrix_index: np.ndarray,
) -> None:
    _patch_matrix_sample_index(monkeypatch, many_matrix_index)

    split_path = resolve_or_create_pinned_split(
        tmp_path, test_ratio=_TEST_RATIO, val_ratio=_VAL_RATIO, seed=_SEED
    )
    actual = json.loads(split_path.read_text())

    fold_by_row: dict[int, str] = {}
    for fold in ("train", "validation", "test"):
        for row in actual[fold]:
            fold_by_row[row] = fold
    for row in actual.get("predict") or []:
        fold_by_row[row] = "predict"

    for matrix_id in np.unique(many_matrix_index):
        rows = np.flatnonzero(many_matrix_index == matrix_id)
        folds = {fold_by_row[int(row)] for row in rows}
        assert len(folds) == 1, f"matrix {matrix_id} rows span multiple folds: {folds}"


def test_resolve_or_create_pinned_split_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    many_matrix_index: np.ndarray,
) -> None:
    _patch_matrix_sample_index(monkeypatch, many_matrix_index)

    first_path = resolve_or_create_pinned_split(
        tmp_path, test_ratio=_TEST_RATIO, val_ratio=_VAL_RATIO, seed=_SEED
    )
    first_content = first_path.read_bytes()

    def _fail_if_called(_dataset_dir: Path) -> np.ndarray:
        raise AssertionError("matrix_sample_index must not be reloaded for an existing split")

    monkeypatch.setattr(_split_pinning, "load_matrix_sample_index", _fail_if_called)

    second_path = resolve_or_create_pinned_split(
        tmp_path, test_ratio=_TEST_RATIO, val_ratio=_VAL_RATIO, seed=_SEED
    )

    assert second_path == first_path
    assert second_path.read_bytes() == first_content
