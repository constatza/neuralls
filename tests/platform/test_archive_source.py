"""The filesystem archive source and its in-memory fake answer every query alike."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tests.fakes.archive_source import InMemoryArchiveSource

from neuralls.domain.generation.ports import ArchiveSource
from neuralls.platform.storage.archive_source import FsArchiveSource

_VECTOR = np.array([1.0, 2.0, 3.0])
_STACK = np.arange(8.0).reshape(4, 2)
_CUBE = np.zeros((2, 2, 2))


def _filesystem(root: Path) -> ArchiveSource:
    (root / "sub").mkdir(parents=True)
    np.save(root / "vec.npy", _VECTOR)
    np.save(root / "stack.npy", _STACK)
    np.save(root / "cube.npy", _CUBE)
    np.savetxt(root / "vec.txt", _VECTOR)
    np.savetxt(root / "sub" / "nested.txt", _VECTOR)
    (root / "data.csv").write_text("1,2\n", encoding="utf-8")
    return FsArchiveSource()


def _in_memory(root: Path) -> ArchiveSource:
    source = InMemoryArchiveSource()
    source.add(root / "vec.npy", _VECTOR)
    source.add(root / "stack.npy", _STACK)
    source.add(root / "cube.npy", _CUBE)
    source.add(root / "vec.txt", _VECTOR)
    source.add(root / "sub" / "nested.txt", _VECTOR)
    source.add(root / "data.csv", _VECTOR)
    return source


@pytest.fixture(params=[_filesystem, _in_memory], ids=["filesystem", "in_memory"])
def source(request: pytest.FixtureRequest, tmp_path: Path) -> ArchiveSource:
    build: Callable[[Path], ArchiveSource] = request.param
    return build(tmp_path)


def test_exists_and_is_file_tell_files_from_directories(source: ArchiveSource, tmp_path: Path):
    assert (
        source.exists(tmp_path / "vec.npy"),
        source.is_file(tmp_path / "vec.npy"),
        source.exists(tmp_path / "sub"),
        source.is_file(tmp_path / "sub"),
        source.exists(tmp_path / "missing.npy"),
    ) == (True, True, True, False, False)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("*.npy", ["cube.npy", "stack.npy", "vec.npy"]), ("**/*.txt", ["sub/nested.txt", "vec.txt"])],
)
def test_glob_returns_sorted_matches(
    source: ArchiveSource, tmp_path: Path, pattern: str, expected: list[str]
):
    assert source.glob(tmp_path, pattern) == [tmp_path / name for name in expected]


@pytest.mark.parametrize(("name", "rows"), [("vec.npy", 1), ("stack.npy", 4), ("vec.txt", 1)])
def test_row_count_follows_the_format(source: ArchiveSource, tmp_path: Path, name: str, rows: int):
    assert source.row_count(tmp_path / name) == rows


@pytest.mark.parametrize("name", ["cube.npy", "data.csv"])
def test_row_count_rejects_unsupported_files(source: ArchiveSource, tmp_path: Path, name: str):
    with pytest.raises(ValueError):
        source.row_count(tmp_path / name)


def test_load_vector_returns_the_stored_values(source: ArchiveSource, tmp_path: Path):
    np.testing.assert_array_equal(source.load_vector(tmp_path / "vec.txt"), _VECTOR)
