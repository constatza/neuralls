"""In-memory ``ArchiveSource`` and matrix readers: the same contract as the filesystem pair.

Files are registered as arrays under paths that never touch disk. ``readers()`` reads the
same registered arrays, so a generation run can be assembled without writing a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.domain.generation.file_sources import MatrixReaders

type Contents = NDArray | csr_array


@dataclass(eq=False, slots=True)
class InMemoryArchiveSource:
    """Archive files held in memory.

    Attributes:
        contents: Array stored at each registered path.
        mtimes: Modification time of each registered path.
        ctimes: Creation time of each path that records one; the others raise like Linux.
    """

    contents: dict[Path, Contents] = field(default_factory=dict)
    mtimes: dict[Path, float] = field(default_factory=dict)
    ctimes: dict[Path, float] = field(default_factory=dict)

    def add(
        self,
        path: Path | str,
        contents: Contents,
        *,
        mtime: float = 0.0,
        ctime: float | None = None,
    ) -> Path:
        """Register ``contents`` at ``path`` and return the path."""
        registered = Path(path)
        self.contents[registered] = contents
        self.mtimes[registered] = mtime
        if ctime is not None:
            self.ctimes[registered] = ctime
        return registered

    def readers(self) -> MatrixReaders:
        """Readers over the registered arrays."""
        return MatrixReaders(generic=self._read, dense=self._read_dense)

    def _read(self, path: Path) -> Contents:
        return self.contents[path]

    def _read_dense(self, path: Path) -> NDArray:
        contents = self.contents[path]
        if isinstance(contents, csr_array):
            raise TypeError(f"{path} holds a sparse array, not a dense .npy array")
        return contents

    def exists(self, path: Path) -> bool:
        return path in self.contents or any(path in file.parents for file in self.contents)

    def is_file(self, path: Path) -> bool:
        return path in self.contents

    def glob(self, root: Path, pattern: str) -> list[Path]:
        return sorted(
            file
            for file in self.contents
            if root in file.parents and file.relative_to(root).full_match(pattern)
        )

    def modified_time(self, path: Path) -> float:
        return self.mtimes[path]

    def creation_time(self, path: Path) -> float:
        try:
            return self.ctimes[path]
        except KeyError:
            raise ValueError(f"enumerate_by='ctime' needs file creation times ({path}).") from None

    def row_count(self, path: Path) -> int:
        match path.suffix:
            case ".npy":
                shape = self.contents[path].shape
                if len(shape) not in (1, 2):
                    raise ValueError(
                        f"Solution .npy file {path} must have shape (n,) or (N,n), got {shape}."
                    )
                return 1 if len(shape) == 1 else int(shape[0])
            case ".txt":
                return 1
            case _:
                raise ValueError(
                    f"Cannot read the row count of solution file {path} "
                    f"(format '{path.suffix}'): only .npy and .txt are supported."
                )

    def load_vector(self, path: Path) -> NDArray:
        return np.asarray(self._read_dense(path))
