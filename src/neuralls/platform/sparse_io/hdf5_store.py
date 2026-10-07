"""hdf5 backend for CSR sample batches.

A batch is one group inside the file at ``SparseLocation.path``. The group's
members use the same names as the zarr layout (see ``layout.py``), so the
logical schema is one document:

- ``MANY_MATRICES``: ``indptr``, ``indices``, ``data`` concatenated over
  samples, ``sample_offsets`` (N+1, nnz boundaries) and ``shape`` (N, 2).
- ``SHARED_PATTERN``: ``indptr`` and ``indices`` stored once, ``data`` as
  (N, nnz), and ``shape`` (N, 2). Absence of ``sample_offsets`` marks this layout.

Datasets are uncompressed with default chunking so the stored values are
bit-identical to the input arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import h5py
import numpy as np
from numpy.typing import NDArray

from neuralls.platform.sparse_io.components import CsrComponents, to_scipy
from neuralls.platform.sparse_io.layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
    choose_layout,
    indptr_offsets,
    members_for,
)
from neuralls.platform.sparse_io.protocol import (
    GROWABLE_CHUNK_ELEMENTS,
    SparseLocation,
    SparseSummary,
)
from neuralls.shared.types import LayoutType

HDF5_GROUP_KEY: Final = "matrix"
"""Group name used when a location carries no explicit key."""

_OPEN_MODE_READ_WRITE: Final = "a"


class Hdf5SparseWriter:
    """Writes a batch of CSR samples as one hdf5 group, replacing any existing group."""

    def write(self, location: SparseLocation, samples: Sequence[CsrComponents]) -> SparseSummary:
        if not samples:
            raise ValueError("cannot write an empty sample batch")
        matrices = [to_scipy(sample) for sample in samples]
        layout = choose_layout(matrices)
        members = members_for(layout, matrices)
        key = _group_key(location)
        location.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(location.path, _OPEN_MODE_READ_WRITE) as file:
            if key in file:
                del file[key]
            group = file.create_group(key)
            for name, values in members.items():
                group.create_dataset(name, data=values)
        return SparseSummary(sample_count=len(samples), layout=layout)


def _group_key(location: SparseLocation) -> str:
    return location.key if location.key is not None else HDF5_GROUP_KEY


class Hdf5SparseReader:
    """Reads CSR samples from a group written by ``Hdf5SparseWriter``."""

    def summary(self, location: SparseLocation) -> SparseSummary:
        with h5py.File(location.path, "r") as file:
            group = file[_group_key(location)]
            sample_count = int(group[SHAPE_ARRAY].shape[0])
            layout = (
                LayoutType.MANY_MATRICES
                if SAMPLE_OFFSETS_ARRAY in group
                else LayoutType.SHARED_PATTERN
            )
        return SparseSummary(sample_count=sample_count, layout=layout)

    def read_sample(self, location: SparseLocation, sample_index: int) -> CsrComponents:
        with h5py.File(location.path, "r") as file:
            group = file[_group_key(location)]
            shapes = group[SHAPE_ARRAY][()]
            sample_count = int(shapes.shape[0])
            if not 0 <= sample_index < sample_count:
                raise IndexError(
                    f"sample_index {sample_index} out of range for {sample_count} samples"
                )
            rows, cols = (int(extent) for extent in shapes[sample_index])
            if SAMPLE_OFFSETS_ARRAY in group:
                return _read_per_sample(group, shapes, sample_index, (rows, cols))
            return _read_shared_pattern(group, sample_index, (rows, cols))


def _read_per_sample(
    group: h5py.Group,
    shapes: NDArray[np.int64],
    sample_index: int,
    shape: tuple[int, int],
) -> CsrComponents:
    offsets = group[SAMPLE_OFFSETS_ARRAY]
    nnz_start, nnz_stop = (int(bound) for bound in offsets[sample_index : sample_index + 2])
    indptr_start, indptr_stop = (
        int(bound) for bound in indptr_offsets(shapes)[sample_index : sample_index + 2]
    )
    return CsrComponents(
        indptr=group[INDPTR_ARRAY][indptr_start:indptr_stop],
        indices=group[INDICES_ARRAY][nnz_start:nnz_stop],
        data=group[DATA_ARRAY][nnz_start:nnz_stop],
        shape=shape,
    )


def _read_shared_pattern(
    group: h5py.Group, sample_index: int, shape: tuple[int, int]
) -> CsrComponents:
    return CsrComponents(
        indptr=group[INDPTR_ARRAY][()],
        indices=group[INDICES_ARRAY][()],
        data=group[DATA_ARRAY][sample_index],
        shape=shape,
    )


class Hdf5MemberSink:
    """Streams members into one group of an hdf5 file, replacing any existing group.

    The file stays open until ``close``, so a streamed dataset is one open handle
    for its whole lifetime rather than one open per batch.
    """

    def __init__(self, path: Path, key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(path, _OPEN_MODE_READ_WRITE)
        if key in self._file:
            del self._file[key]
        self._group = self._file.create_group(key)

    def create_fixed(self, name: str, shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> None:
        self._group.create_dataset(name, shape=shape, dtype=dtype)

    def create_growable(self, name: str, dtype: np.dtype[np.generic]) -> None:
        self._group.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(GROWABLE_CHUNK_ELEMENTS,),
        )

    def write_rows(self, name: str, start: int, values: NDArray[np.generic]) -> None:
        self._group[name][start : start + values.shape[0]] = values

    def append(self, name: str, values: NDArray[np.generic]) -> None:
        dataset = self._group[name]
        old_length = int(dataset.shape[0])
        dataset.resize((old_length + values.shape[0],))
        dataset[old_length:] = values

    def close(self) -> None:
        self._file.close()
