"""zarr backend for CSR sample batches.

A batch is one zarr group at ``SparseLocation.path``. The group's members use
the shared schema in ``layout.py``, so a zarr batch and an hdf5 batch of the
same matrices differ only in container. ``SparseLocation.key`` is not used:
the group directory is the location.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from neuralls.platform.sparse_io.components import INDEX_DTYPE, VALUE_DTYPE, CsrComponents, to_scipy
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


class ZarrSparseWriter:
    """Writes a batch of CSR samples as one zarr group, replacing any existing group."""

    def write(self, location: SparseLocation, samples: Sequence[CsrComponents]) -> SparseSummary:
        if not samples:
            raise ValueError("cannot write an empty sample batch")
        matrices = [to_scipy(sample) for sample in samples]
        layout = choose_layout(matrices)
        group = zarr.open_group(str(location.path), mode="w")
        for name, values in members_for(layout, matrices).items():
            group.create_array(name, data=values)
        return SparseSummary(sample_count=len(samples), layout=layout)


def _layout_of(group: zarr.Group) -> LayoutType:
    """Only the per-sample layout stores offsets, so their presence identifies it."""
    return (
        LayoutType.MANY_MATRICES
        if SAMPLE_OFFSETS_ARRAY in group.array_keys()
        else LayoutType.SHARED_PATTERN
    )


def _member(group: zarr.Group, name: str) -> zarr.Array:
    """Return one storage member, rejecting a group where an array is required."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        raise TypeError(f"CSR storage member {name!r} must be a zarr array")
    return member


def _int_array(member: zarr.Array, selection: slice | tuple[()] = ()) -> NDArray[np.int64]:
    return np.asarray(member[selection], dtype=INDEX_DTYPE)


class ZarrSparseReader:
    """Reads CSR samples from a group written by ``ZarrSparseWriter``."""

    def summary(self, location: SparseLocation) -> SparseSummary:
        group = zarr.open_group(str(location.path), mode="r")
        sample_count = int(_member(group, SHAPE_ARRAY).shape[0])
        return SparseSummary(sample_count=sample_count, layout=_layout_of(group))

    def read_sample(self, location: SparseLocation, sample_index: int) -> CsrComponents:
        """Read one sample, touching only that sample's slices of the members.

        The shape table and nnz offsets are small and read whole. The shared
        pattern is read whole and only the sample's data row is sliced.
        """
        group = zarr.open_group(str(location.path), mode="r")
        shape_table = _int_array(_member(group, SHAPE_ARRAY))
        sample_count = int(shape_table.shape[0])
        if not 0 <= sample_index < sample_count:
            raise IndexError(f"sample_index {sample_index} out of range for {sample_count} samples")
        rows, cols = (int(extent) for extent in shape_table[sample_index])
        if _layout_of(group) is LayoutType.SHARED_PATTERN:
            indptr = _int_array(_member(group, INDPTR_ARRAY))
            indices = _int_array(_member(group, INDICES_ARRAY))
            data = np.asarray(_member(group, DATA_ARRAY)[sample_index], dtype=VALUE_DTYPE)
        else:
            ptr_bounds = indptr_offsets(shape_table)
            nnz_bounds = _int_array(_member(group, SAMPLE_OFFSETS_ARRAY))
            ptr_start, ptr_stop = (
                int(bound) for bound in ptr_bounds[sample_index : sample_index + 2]
            )
            nnz_start, nnz_stop = (
                int(bound) for bound in nnz_bounds[sample_index : sample_index + 2]
            )
            indptr = _int_array(_member(group, INDPTR_ARRAY), slice(ptr_start, ptr_stop))
            indices = _int_array(_member(group, INDICES_ARRAY), slice(nnz_start, nnz_stop))
            data = np.asarray(_member(group, DATA_ARRAY)[nnz_start:nnz_stop], dtype=VALUE_DTYPE)
        return CsrComponents(indptr=indptr, indices=indices, data=data, shape=(rows, cols))


class ZarrMemberSink:
    """Streams members into a zarr group, replacing any existing group at the location."""

    def __init__(self, path: Path) -> None:
        self._group = zarr.open_group(str(path), mode="w")

    def create_fixed(self, name: str, shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> None:
        self._group.create_array(name, shape=shape, dtype=dtype)

    def create_growable(self, name: str, dtype: np.dtype[np.generic]) -> None:
        self._group.create_array(name, shape=(0,), dtype=dtype, chunks=(GROWABLE_CHUNK_ELEMENTS,))

    def write_rows(self, name: str, start: int, values: NDArray[np.generic]) -> None:
        member = _member(self._group, name)
        member[start : start + values.shape[0]] = values

    def append(self, name: str, values: NDArray[np.generic]) -> None:
        member = _member(self._group, name)
        old_length = int(member.shape[0])
        member.resize((old_length + values.shape[0],))
        member[old_length:] = values

    def close(self) -> None:
        """Nothing to flush: zarr writes each chunk when it is assigned."""
