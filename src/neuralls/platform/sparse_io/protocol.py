"""Storage-agnostic contracts for reading and writing batches of CSR samples.

Backends implement these protocols structurally, so the composition layer can
depend on the contract without importing any particular storage library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import CsrComponents
from neuralls.shared.types import LayoutType


@dataclass(frozen=True)
class SparseLocation:
    """Where a batch of samples lives: a file or store path, plus an optional member key."""

    path: Path
    key: str | None = None


@dataclass(frozen=True)
class SparseSummary:
    """Sample count and storage layout of a stored batch."""

    sample_count: int
    layout: LayoutType


class SparseReader(Protocol):
    """Reads a stored batch of CSR samples one at a time."""

    def summary(self, location: SparseLocation) -> SparseSummary: ...

    def read_sample(self, location: SparseLocation, sample_index: int) -> CsrComponents: ...


class SparseWriter(Protocol):
    """Writes a batch of CSR samples to a location."""

    def write(
        self, location: SparseLocation, samples: Sequence[CsrComponents]
    ) -> SparseSummary: ...


GROWABLE_CHUNK_ELEMENTS: Final = 4096
"""Chunk length of growable CSR members: appends extend the array in chunks of this many elements."""


class CsrMemberSink(Protocol):
    """Write-only access to the named members of one CSR container while it is streamed.

    Fixed members are created at their full shape and filled by row offset. Growable
    members start empty and are extended at the end. Backends implement this for
    zarr and hdf5; the stream writer owns the offsets.
    """

    def create_fixed(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype[np.generic]
    ) -> None: ...

    def create_growable(self, name: str, dtype: np.dtype[np.generic]) -> None: ...

    def write_rows(self, name: str, start: int, values: NDArray[np.generic]) -> None: ...

    def append(self, name: str, values: NDArray[np.generic]) -> None: ...

    def close(self) -> None: ...


class SparseStreamWriter(Protocol):
    """Writes CSR samples incrementally, one batch at a time, and never reads them back.

    The number of samples and the layout are fixed when the writer is opened, so the
    stored arrays can be sized before their contents are known.
    """

    def write_batch(self, samples: Sequence[csr_array]) -> None: ...

    def close(self) -> SparseSummary:
        """Verify that exactly the planned number of samples was written and finish the container."""
        ...
