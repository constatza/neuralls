"""Pre-sized dense array stores that accept rows at explicit offsets.

Streaming generation knows the total row count N before any data exists, so each
array is created at its full shape and filled batch by batch. Creating at full
shape (instead of growing with resize) keeps the on-disk layout identical to the
buffered accumulators, and lets close() detect a source that under-produced rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import zarr
from numpy.typing import NDArray

from neuralls.domain.generation.ports import ArrayStore
from neuralls.platform.storage.errors import _raise_storage_error

# Chunking mirrors the buffered dense accumulators: one row per chunk along the sample axis.
ROW_CHUNK_ROWS: int = 1


def _chunks_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Chunk shape for an array: one row per chunk for matrices and vectors of rows.

    A one-dimensional array (row_kind, matrix_sample_index) is a single chunk. One chunk
    per row would create one file per sample in zarr, which the buffered writers never did.
    """
    if len(shape) == 1:
        return shape
    return (ROW_CHUNK_ROWS, *shape[1:])


__all__ = ["ArrayStore", "Hdf5ArrayStore", "ZarrArrayStore"]


@dataclass
class _ArrayLedger:
    """Shape and per-row written flags for one named array."""

    shape: tuple[int, ...]
    written: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        self.written = np.zeros(self.shape[0], dtype=np.bool_)


class _RowLedger:
    """Tracks which rows of each array were written and validates every operation.

    Shared by all backends so the guard clauses and completeness rule are defined once.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, _ArrayLedger] = {}

    def reserve(self, name: str, shape: tuple[int, ...]) -> None:
        if name in self._arrays:
            raise ValueError(f"array {name!r} already exists in this store")
        if not shape or any(dim < 0 for dim in shape):
            raise ValueError(
                f"array {name!r} needs a non-empty shape of non-negative dims, got {shape}"
            )
        self._arrays[name] = _ArrayLedger(shape)

    def validate_write(self, name: str, start: int, data: NDArray) -> int:
        """Check a write and return its row count. Raises ValueError on any violation."""
        ledger = self._lookup(name)
        expected_shape = ledger.shape
        if data.ndim != len(expected_shape) or data.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"write to {name!r} has shape {data.shape}, expected trailing shape {expected_shape[1:]}"
            )
        if start < 0:
            raise ValueError(f"write to {name!r} starts at negative row {start}")
        rows = int(data.shape[0])
        if start + rows > expected_shape[0]:
            raise ValueError(
                f"write to {name!r} covers rows [{start}, {start + rows}) beyond the "
                f"{expected_shape[0]} rows it was created with"
            )
        return rows

    def mark_written(self, name: str, start: int, rows: int) -> None:
        self._lookup(name).written[start : start + rows] = True

    def raise_if_incomplete(self) -> None:
        problems = [
            f"array {name!r} expects {ledger.shape[0]} rows but {int(ledger.written.sum())} were written"
            for name, ledger in self._arrays.items()
            if int(ledger.written.sum()) != ledger.shape[0]
        ]
        if problems:
            raise ValueError("; ".join(problems))

    def _lookup(self, name: str) -> _ArrayLedger:
        ledger = self._arrays.get(name)
        if ledger is None:
            raise ValueError(f"unknown array {name!r}; create it before writing")
        return ledger


class ZarrArrayStore:
    """One zarr array per name, all inside a single zarr group at `root`."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._ledger = _RowLedger()
        self._arrays: dict[str, zarr.Array] = {}
        try:
            self._group = zarr.open_group(str(self._root), mode="w")
        except OSError as exc:
            _raise_storage_error("Opening dense zarr array store", self._root, exc)

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self._ledger.reserve(name, shape)
        try:
            self._arrays[name] = self._group.create_array(
                name,
                shape=shape,
                dtype=dtype,
                chunks=_chunks_for(shape),
            )
        except OSError as exc:
            _raise_storage_error(f"Creating zarr array {name!r}", self._root, exc)

    def write_rows(self, name: str, start: int, data: NDArray) -> None:
        rows = self._ledger.validate_write(name, start, data)
        try:
            self._arrays[name][start : start + rows] = data
        except OSError as exc:
            _raise_storage_error(f"Writing zarr array {name!r}", self._root, exc)
        self._ledger.mark_written(name, start, rows)

    def close(self) -> None:
        self._ledger.raise_if_incomplete()


class Hdf5ArrayStore:
    """One HDF5 dataset per name, all inside a single file at `path`."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._ledger = _RowLedger()
        self._datasets: dict[str, h5py.Dataset] = {}
        try:
            self._file = h5py.File(str(self._path), "w")
        except OSError as exc:
            _raise_storage_error("Opening dense HDF5 array store", self._path, exc)

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self._ledger.reserve(name, shape)
        try:
            self._datasets[name] = self._file.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                chunks=_chunks_for(shape),
            )
        except OSError as exc:
            _raise_storage_error(f"Creating HDF5 dataset {name!r}", self._path, exc)

    def write_rows(self, name: str, start: int, data: NDArray) -> None:
        rows = self._ledger.validate_write(name, start, data)
        try:
            self._datasets[name][start : start + rows] = data
        except OSError as exc:
            _raise_storage_error(f"Writing HDF5 dataset {name!r}", self._path, exc)
        self._ledger.mark_written(name, start, rows)

    def close(self) -> None:
        try:
            self._ledger.raise_if_incomplete()
        finally:
            self._file.close()
