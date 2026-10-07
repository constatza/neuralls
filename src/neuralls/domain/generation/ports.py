"""Ports used by generation workflows."""

from __future__ import annotations

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
