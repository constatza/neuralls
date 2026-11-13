"""Filesystem adapter for the files archive-backed generation draws from.

I/O actions - these functions read from disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuralls.domain.generation.ports import ArchiveSource


@dataclass(frozen=True, slots=True)
class FsArchiveSource:
    """Reads archive files straight off the local filesystem."""

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def glob(self, root: Path, pattern: str) -> list[Path]:
        return sorted(root.glob(pattern))

    def modified_time(self, path: Path) -> float:
        return path.stat().st_mtime

    def creation_time(self, path: Path) -> float:
        try:
            return path.stat().st_birthtime
        except AttributeError:
            raise ValueError(
                f"enumerate_by='ctime' needs file creation times, which this platform does not "
                f"record ({path}). Use enumerate_by='mtime' or 'name', or sample_id_regex."
            ) from None

    def row_count(self, path: Path) -> int:
        """Rows of a solution file: .npy from the array header (1-D is one sample), .txt as one.

        A .txt file is one sample whatever its line count: ``load_vector`` loads the whole
        file as a single vector, so counting lines would promise rows it never yields.
        """
        match path.suffix:
            case ".npy":
                shape = _npy_shape(path)
                match len(shape):
                    case 1:
                        return 1
                    case 2:
                        return int(shape[0])
                    case _:
                        raise ValueError(
                            f"Solution .npy file {path} must have shape (n,) or (N,n), got {shape}."
                        )
            case ".txt":
                return 1
            case _:
                raise ValueError(
                    f"Cannot read the row count of solution file {path} "
                    f"(format '{path.suffix}'): only .npy and .txt are supported."
                )

    def load_vector(self, path: Path) -> NDArray:
        return np.loadtxt(path)


def _npy_shape(path: Path) -> tuple[int, ...]:
    """The array shape from a .npy header, without reading the data."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported .npy version {version} in solution file {path}.")
    return shape


@dataclass(frozen=True, eq=False, slots=True)
class CachingArchiveSource:
    """An ``ArchiveSource`` that loads each vector once and serves repeats from memory.

    Build one per generation run: every binding of a one-matrix source draws the same
    files, and the cache lives exactly as long as the run that holds this source, so no
    loaded vector outlives it. Nothing is evicted: peak memory is every distinct file the
    run loads. Cached arrays are read-only, so a caller that mutates one fails instead of
    corrupting later loads. Listing, timestamps and row counts pass through uncached.
    """

    inner: ArchiveSource
    _vectors: dict[Path, NDArray] = field(default_factory=dict, init=False)

    def exists(self, path: Path) -> bool:
        return self.inner.exists(path)

    def is_file(self, path: Path) -> bool:
        return self.inner.is_file(path)

    def glob(self, root: Path, pattern: str) -> list[Path]:
        return self.inner.glob(root, pattern)

    def modified_time(self, path: Path) -> float:
        return self.inner.modified_time(path)

    def creation_time(self, path: Path) -> float:
        return self.inner.creation_time(path)

    def row_count(self, path: Path) -> int:
        return self.inner.row_count(path)

    def load_vector(self, path: Path) -> NDArray:
        if path not in self._vectors:
            vector = self.inner.load_vector(path)
            vector.flags.writeable = False
            self._vectors[path] = vector
        return self._vectors[path]
