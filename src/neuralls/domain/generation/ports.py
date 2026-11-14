"""Ports used by generation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray


class TracingSolverPort(Protocol):
    """Callable protocol for traced linear solves."""

    def __call__(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[NDArray, Any]: ...


class ArrayStore(Protocol):
    """Write-only sink for arrays whose row count is fixed at creation.

    Defined in the domain so generation can depend on it without importing platform
    storage; each backend in platform/storage satisfies it structurally.
    """

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        """Create a named array at its full shape. Raises ValueError if the name exists."""
        ...

    def write_rows(self, name: str, start: int, data: NDArray) -> None:
        """Write `data` into rows [start, start + len(data)) of the named array."""
        ...

    def close(self) -> None:
        """Flush and release the store. Raises ValueError if any array is incomplete."""
        ...


class ArchiveSource(Protocol):
    """Read-only view of the files archive-backed generation draws from.

    Defined in the domain so file selection, sample-id derivation and vector loading never
    touch the filesystem; ``platform/storage/archive_source.py`` satisfies it with the real
    one and tests satisfy it with an in-memory fake. Caching repeated loads is the
    source's own concern (``CachingArchiveSource``), never the domain's.
    """

    def exists(self, path: Path) -> bool:
        """Whether ``path`` exists."""
        ...

    def is_file(self, path: Path) -> bool:
        """Whether ``path`` is an existing regular file."""
        ...

    def glob(self, root: Path, pattern: str) -> list[Path]:
        """Paths under ``root`` matching ``pattern`` (recursive ``**`` allowed), sorted."""
        ...

    def modified_time(self, path: Path) -> float:
        """Last-modification timestamp of ``path``."""
        ...

    def creation_time(self, path: Path) -> float:
        """Creation timestamp of ``path``.

        Raises:
            ValueError: The platform records no creation time. Falling back to the
                modification time would give one TOML different sample ids per OS.
        """
        ...

    def row_count(self, path: Path) -> int:
        """Rows of an explicit solution file, without reading its values.

        Raises:
            ValueError: The format has no cheap row count, or the array rank is unsupported.
        """
        ...

    def load_vector(self, path: Path) -> NDArray:
        """Load the whole file at ``path`` as one vector."""
        ...
