"""Filesystem adapter for the files archive-backed generation draws from.

I/O actions - these functions read from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


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
