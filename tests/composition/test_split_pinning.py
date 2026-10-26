"""Generation pins a deterministic, matrix-group-aware split and rewrites it only on change."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.types.split import RatioSplitStrategy

from neuralls.composition.generation import split_pinning
from neuralls.composition.generation.split_pinning import pin_dataset_split
from neuralls.platform.config.models.data_models import SplitConfig

_CFG = SplitConfig(test_ratio=0.2, val_ratio=0.2, seed=20240521)


@pytest.fixture
def single_matrix_index() -> np.ndarray:
    return np.zeros(10, dtype=np.int64)


@pytest.fixture
def many_matrix_index() -> np.ndarray:
    # Matrix 0 has 3 rows, matrix 1 has 2 rows, matrix 2 has 1 row.
    return np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)


def _bind(monkeypatch: pytest.MonkeyPatch, values: np.ndarray) -> None:
    monkeypatch.setattr(split_pinning, "load_matrix_sample_index", lambda _dir: values)


def test_single_matrix_delegates_to_ratio_split_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single_matrix_index: np.ndarray
) -> None:
    _bind(monkeypatch, single_matrix_index)

    split_path = pin_dataset_split(tmp_path, _CFG)

    expected = RatioSplitStrategy(
        num_samples=len(single_matrix_index), test_ratio=0.2, val_ratio=0.2, seed=_CFG.seed
    ).split()
    actual = json.loads(split_path.read_text())
    assert sorted(actual["train"]) == sorted(expected.train)
    assert sorted(actual["validation"]) == sorted(expected.validation)
    assert sorted(actual["test"]) == sorted(expected.test)


def test_many_matrix_keeps_each_matrixs_rows_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, many_matrix_index: np.ndarray
) -> None:
    _bind(monkeypatch, many_matrix_index)

    actual = json.loads(pin_dataset_split(tmp_path, _CFG).read_text())

    fold_by_row = {
        row: fold for fold in ("train", "validation", "test") for row in actual[fold]
    } | {row: "predict" for row in actual.get("predict") or []}
    for matrix_id in np.unique(many_matrix_index):
        rows = np.flatnonzero(many_matrix_index == matrix_id)
        assert len({fold_by_row[int(row)] for row in rows}) == 1


def test_unchanged_split_leaves_the_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, many_matrix_index: np.ndarray
) -> None:
    _bind(monkeypatch, many_matrix_index)
    first = pin_dataset_split(tmp_path, _CFG)
    first_mtime = first.stat().st_mtime_ns

    second = pin_dataset_split(tmp_path, _CFG)

    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


def test_changed_config_replaces_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, many_matrix_index: np.ndarray
) -> None:
    _bind(monkeypatch, many_matrix_index)
    first_bytes = pin_dataset_split(tmp_path, _CFG).read_bytes()

    replaced = pin_dataset_split(tmp_path, SplitConfig(test_ratio=0.5, val_ratio=0.0, seed=1))

    assert replaced.read_bytes() != first_bytes


def test_ratios_that_exceed_the_dataset_are_rejected() -> None:
    with pytest.raises(ValueError):
        SplitConfig(test_ratio=0.7, val_ratio=0.4)
