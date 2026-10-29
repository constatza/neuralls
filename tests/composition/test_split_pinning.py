"""Generation pins a deterministic, matrix-group-aware split and rewrites it only on change."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.types.split import IndexSplit, RatioSplitStrategy

from neuralls.composition.generation.split_pinning import pin_split, plan_split, split_is_current
from neuralls.platform.config.models.data_models import SplitConfig
from tests.fakes.split_store import InMemorySplitStore

_CFG = SplitConfig(test_ratio=0.2, val_ratio=0.2, seed=20240521)
_DATASET = Path("dataset")


@pytest.fixture
def single_matrix_index() -> np.ndarray:
    return np.zeros(10, dtype=np.int64)


@pytest.fixture
def many_matrix_index() -> np.ndarray:
    # Matrix 0 has 3 rows, matrix 1 has 2 rows, matrix 2 has 1 row.
    return np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)


def test_single_matrix_delegates_to_ratio_split_strategy(single_matrix_index: np.ndarray) -> None:
    planned = plan_split(single_matrix_index, _CFG)

    expected = RatioSplitStrategy(
        num_samples=len(single_matrix_index), test_ratio=0.2, val_ratio=0.2, seed=_CFG.seed
    ).split()
    assert sorted(planned.train) == sorted(expected.train)
    assert sorted(planned.validation) == sorted(expected.validation)
    assert sorted(planned.test) == sorted(expected.test)


def test_many_matrix_keeps_each_matrixs_rows_together(many_matrix_index: np.ndarray) -> None:
    planned = plan_split(many_matrix_index, _CFG)

    fold_by_row = {
        row: fold
        for fold, rows in (
            ("train", planned.train),
            ("validation", planned.validation),
            ("test", planned.test),
            ("predict", planned.predict or ()),
        )
        for row in rows
    }
    for matrix_id in np.unique(many_matrix_index):
        rows = np.flatnonzero(many_matrix_index == matrix_id)
        assert len({fold_by_row[int(row)] for row in rows}) == 1


def test_the_plan_is_deterministic(many_matrix_index: np.ndarray) -> None:
    assert plan_split(many_matrix_index, _CFG) == plan_split(many_matrix_index, _CFG)


def test_pinning_a_new_split_writes_it(many_matrix_index: np.ndarray) -> None:
    store = InMemorySplitStore()
    split = plan_split(many_matrix_index, _CFG)

    pin_split(_DATASET, split, _CFG, store)

    assert store.read(_DATASET) == split
    assert store.writes == [_DATASET]


def test_unchanged_split_is_not_rewritten(many_matrix_index: np.ndarray) -> None:
    store = InMemorySplitStore()
    split = plan_split(many_matrix_index, _CFG)
    first = pin_split(_DATASET, split, _CFG, store)

    second = pin_split(_DATASET, split, _CFG, store)

    assert second == first
    assert store.writes == [_DATASET]


def test_changed_config_replaces_the_split(many_matrix_index: np.ndarray) -> None:
    store = InMemorySplitStore()
    pin_split(_DATASET, plan_split(many_matrix_index, _CFG), _CFG, store)
    other = SplitConfig(test_ratio=0.5, val_ratio=0.0, seed=1)
    replaced = plan_split(many_matrix_index, other)

    pin_split(_DATASET, replaced, other, store)

    assert store.read(_DATASET) == replaced
    assert store.writes == [_DATASET, _DATASET]


def test_a_split_is_current_only_when_it_is_pinned_and_equal(
    many_matrix_index: np.ndarray,
) -> None:
    store = InMemorySplitStore()
    split = plan_split(many_matrix_index, _CFG)
    stale = IndexSplit(train=(0,), validation=(1,), test=(2,), predict=None)

    assert not split_is_current(_DATASET, split, store)
    store.write(_DATASET, stale)
    assert not split_is_current(_DATASET, split, store)
    store.write(_DATASET, split)
    assert split_is_current(_DATASET, split, store)


def test_ratios_that_exceed_the_dataset_are_rejected() -> None:
    with pytest.raises(ValueError):
        SplitConfig(test_ratio=0.7, val_ratio=0.4)
