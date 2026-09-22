"""Fixtures for digest tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
import zarr
from pydantic import BaseModel

from neuralls.shared.digest import Cosmetic, InputConfig, InputData


class CosmeticModel(BaseModel):
    """Pydantic model with one identity field and one cosmetic label."""

    value: int = 1
    label: Annotated[str, Cosmetic()] = "a"


class DefaultModel(BaseModel):
    """Pydantic model whose default value feeds identity."""

    threshold: float = 0.0


@dataclass(frozen=True)
class CosmeticDataclass:
    """Dataclass with one identity field and one cosmetic label."""

    value: int = 1
    label: Annotated[str, Cosmetic()] = "a"


class InputModel(BaseModel):
    """Pydantic model with data and config file inputs."""

    data: Annotated[Path | None, InputData()] = None
    config: Annotated[Path | None, InputConfig()] = None


class UnmarkedPathModel(BaseModel):
    """Pydantic model with an unmarked Path (must fail at digest time)."""

    path: Path


class Unknown:
    """A type the canonicalizer does not know."""


type WriteTree = Callable[[Path, dict[str, bytes]], Path]


@pytest.fixture
def tree_files() -> dict[str, bytes]:
    """Relative-path -> bytes mapping used to build directories."""
    return {"a.bin": b"alpha", "nested/b.bin": b"beta", "nested/deep/c.bin": b"gamma"}


@pytest.fixture
def write_tree() -> WriteTree:
    """Return a helper writing a mapping of files under a root."""

    def _write(root: Path, files: dict[str, bytes]) -> Path:
        for rel, payload in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return root

    return _write


@pytest.fixture
def dir_a(tmp_path: Path, tree_files: dict[str, bytes], write_tree: WriteTree) -> Path:
    """Directory holding ``tree_files``."""
    return write_tree(tmp_path / "one" / "data", tree_files)


@pytest.fixture
def dir_b(tmp_path: Path, tree_files: dict[str, bytes], write_tree: WriteTree) -> Path:
    """Identical copy of ``dir_a`` in a different location."""
    return write_tree(tmp_path / "two" / "elsewhere", tree_files)


@pytest.fixture
def toml_main(tmp_path: Path) -> Path:
    """Main TOML file referencing a sibling TOML by relative name."""
    (tmp_path / "child.toml").write_text('lr = 0.1\nname = "child"\n')
    main = tmp_path / "main.toml"
    main.write_text('# comment\nprofile = "child.toml"\nepochs = 3\n')
    return main


@pytest.fixture
def toml_cycle(tmp_path: Path) -> Path:
    """Two TOML files referencing each other."""
    (tmp_path / "x.toml").write_text('next = "y.toml"\n')
    (tmp_path / "y.toml").write_text('next = "x.toml"\n')
    return tmp_path / "x.toml"


@pytest.fixture
def simple_toml(tmp_path: Path) -> Path:
    """Small standalone TOML file."""
    path = tmp_path / "simple.toml"
    path.write_text("x = 1\n")
    return path


@pytest.fixture
def toml_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two TOML files with equal meaning but different comments, spacing and order."""
    first = tmp_path / "a.toml"
    second = tmp_path / "b.toml"
    first.write_text("x = 1\ny = [1, 2]\n[t]\nk = 'v'\n")
    second.write_text("# hello\ny = [ 1,2 ]  # c\nx=1\n\n\n[t]\nk   =   'v'\n")
    return first, second


@pytest.fixture
def sample_array() -> np.ndarray:
    """Seeded float64 array (40 x 6)."""
    return np.random.default_rng(0).standard_normal((40, 6))


@pytest.fixture
def stored_arrays(tmp_path: Path, sample_array: np.ndarray) -> dict[str, object]:
    """The same array persisted as npy (memmap), HDF5 and zarr."""
    npy = tmp_path / "a.npy"
    np.save(npy, sample_array)
    h5_path = tmp_path / "a.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("x", data=sample_array, chunks=(8, 6), compression="gzip")
    zarr_path = tmp_path / "a.zarr"
    zarr.save_array(str(zarr_path), sample_array)
    return {
        "npy": np.load(npy, mmap_mode="r"),
        "h5": h5py.File(h5_path, "r")["x"],
        "zarr": zarr.open_array(str(zarr_path), mode="r"),
    }
