"""`FsSplitStore` and the in-memory fake obey one `SplitStore` contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.types.split import IndexSplit
from tests.fakes.split_store import InMemorySplitStore

from neuralls.composition.generation.ports import SplitStore
from neuralls.platform.storage.split_store import FsSplitStore
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale

_SPLIT = IndexSplit(train=(0, 1, 2), validation=(3,), test=(4,), predict=None)


@pytest.fixture(params=["filesystem", "in-memory"])
def store(request: pytest.FixtureRequest) -> SplitStore:
    return FsSplitStore() if request.param == "filesystem" else InMemorySplitStore()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "dataset"


def test_an_unpinned_dataset_reads_as_none(store: SplitStore, dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    assert store.read(dataset_dir) is None


def test_a_written_split_reads_back_and_is_required_at_the_split_file(
    store: SplitStore, dataset_dir: Path
) -> None:
    dataset_dir.mkdir()

    written = store.write(dataset_dir, _SPLIT)

    assert written == dataset_dir / SPLIT_FILENAME
    assert store.read(dataset_dir) == _SPLIT
    assert store.require(dataset_dir) == written


def test_requiring_an_unpinned_split_asks_for_generation(
    store: SplitStore, dataset_dir: Path
) -> None:
    dataset_dir.mkdir()
    with pytest.raises(StageInputsStale):
        store.require(dataset_dir)
