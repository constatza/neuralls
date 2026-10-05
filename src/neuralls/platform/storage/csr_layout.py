"""On-disk layout of CSR system matrices stored as a zarr group.

This module is the single authority for the CSR storage schema. Each stored
matrix is a zarr group (``matrix/``) holding these members:

Per-sample layout (``LayoutType.MANY_MATRICES``, the default). Every sample has
its own sparsity pattern.
    indptr          int64, (sum_i(rows_i + 1),)  per-sample indptr arrays concatenated.
    indices         int64, (sum_i(nnz_i),)       per-sample column indices concatenated.
    data            float64, (sum_i(nnz_i),)     per-sample values concatenated.
    sample_offsets  int64, (N + 1,)              nnz offsets: sample i owns
                                                 data[sample_offsets[i]:sample_offsets[i+1]].
                                                 The indptr offsets follow from ``shape``:
                                                 sample i owns rows_i + 1 indptr entries.
    shape           int64, (N, 2)                (rows, cols) of every sample.

Broadcast layout (``LayoutType.SHARED_PATTERN``). All samples share one pattern.
    indptr          int64, (rows + 1,)           stored once.
    indices         int64, (nnz,)                stored once.
    data            float64, (N, nnz)            values of sample i are data[i].
    shape           int64, (N, 2)                (rows, cols) of every sample.
    sample_offsets is not written in this layout: every sample has the same nnz.

Indices and indptr are always the scipy CSR convention: ``indptr[0] == 0``,
``indptr[-1] == len(indices) == len(data)`` for one sample, and column indices
are within ``[0, cols)``. Writing the explicit ``shape`` array keeps the reader
from having to infer dimensions from the pattern, which would be ambiguous for
trailing empty rows or columns.

This module is pure: it validates and converts between ``csr_array`` and the
flat arrays above. It performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.shared.types import LayoutType

INDPTR_ARRAY: Final = "indptr"
INDICES_ARRAY: Final = "indices"
DATA_ARRAY: Final = "data"
SAMPLE_OFFSETS_ARRAY: Final = "sample_offsets"
SHAPE_ARRAY: Final = "shape"
"""Member name of the (N, 2) int64 per-sample shape array."""

INDEX_DTYPE: Final = np.int64
VALUE_DTYPE: Final = np.float64
SHAPE_COLUMNS: Final = 2
"""A matrix shape is (rows, cols)."""


@dataclass(frozen=True)
class CsrArrays:
    """The three CSR arrays plus the matrix shape, for one matrix."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    shape: tuple[int, int]


@dataclass(frozen=True)
class PerSamplePack:
    """Flat per-sample arrays for ``LayoutType.MANY_MATRICES``."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    sample_offsets: NDArray[np.int64]
    shape: NDArray[np.int64]


@dataclass(frozen=True)
class SharedPatternPack:
    """Shared-pattern arrays for ``LayoutType.SHARED_PATTERN``."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    shape: NDArray[np.int64]


def build_csr(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    shape: tuple[int, int],
) -> csr_array:
    """Assemble a ``csr_array`` from its three arrays and shape.

    Args:
        indptr: Row pointer array of length ``shape[0] + 1``.
        indices: Column indices, one per stored value.
        data: Stored values.
        shape: ``(rows, cols)`` of the matrix.

    Returns:
        The CSR matrix, with the given arrays used as-is (no copy is forced).

    Raises:
        ValueError: If the lengths are inconsistent or ``indptr[-1] != len(data)``.
    """
    rows = shape[0]
    if indptr.shape != (rows + 1,):
        raise ValueError(f"indptr must have length {rows + 1} for {rows} rows, got {indptr.shape}")
    if indices.shape != data.shape:
        raise ValueError(
            f"indices and data must have equal length, got {indices.shape[0]} and {data.shape[0]}"
        )
    if int(indptr[-1]) != data.shape[0]:
        raise ValueError(
            f"indptr[-1]={int(indptr[-1])} must equal the number of stored values {data.shape[0]}"
        )
    return csr_array((data, indices, indptr), shape=shape)


def split_csr(matrix: csr_array) -> CsrArrays:
    """Split a ``csr_array`` into its three arrays and shape.

    Args:
        matrix: CSR matrix. It is converted to canonical CSR form first, so the
            caller's matrix is not modified.

    Returns:
        The arrays with the integer dtypes of the storage schema.
    """
    canonical = csr_array(matrix, copy=True)
    canonical.sort_indices()
    rows, cols = canonical.shape
    return CsrArrays(
        indptr=canonical.indptr.astype(INDEX_DTYPE, copy=False),
        indices=canonical.indices.astype(INDEX_DTYPE, copy=False),
        data=canonical.data.astype(VALUE_DTYPE, copy=False),
        shape=(int(rows), int(cols)),
    )


def choose_layout(matrices: Sequence[csr_array]) -> LayoutType:
    """Pick the storage layout from the sparsity patterns alone.

    Args:
        matrices: One or more CSR samples.

    Returns:
        ``SHARED_PATTERN`` when every sample has the same shape, indptr and
        indices, otherwise ``MANY_MATRICES``.

    Raises:
        ValueError: If ``matrices`` is empty.
    """
    if not matrices:
        raise ValueError("choose_layout requires at least one matrix")
    first = split_csr(matrices[0])
    for matrix in matrices[1:]:
        other = split_csr(matrix)
        if other.shape != first.shape:
            return LayoutType.MANY_MATRICES
        if not (
            np.array_equal(other.indptr, first.indptr)
            and np.array_equal(other.indices, first.indices)
        ):
            return LayoutType.MANY_MATRICES
    return LayoutType.SHARED_PATTERN


def shape_table(shapes: Sequence[tuple[int, int]]) -> NDArray[np.int64]:
    """Return the (N, 2) int64 shape array for a list of matrix shapes."""
    return np.asarray(shapes, dtype=INDEX_DTYPE).reshape(len(shapes), SHAPE_COLUMNS)


def indptr_offsets(shape_array: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the (N + 1) offsets of each sample's indptr inside the flat indptr array.

    Sample i owns ``rows_i + 1`` indptr entries, so the offsets are the
    cumulative sums of ``rows + 1``.
    """
    row_counts = shape_array[:, 0] + 1
    return np.concatenate(([0], np.cumsum(row_counts))).astype(INDEX_DTYPE)


def pack_per_sample(matrices: Sequence[csr_array]) -> PerSamplePack:
    """Concatenate per-sample CSR arrays into the ``MANY_MATRICES`` layout.

    Args:
        matrices: One or more CSR samples, each with its own pattern.

    Returns:
        The flat arrays and offsets described in the module docstring.
    """
    parts = [split_csr(matrix) for matrix in matrices]
    shape_array = shape_table([part.shape for part in parts])
    nnz_counts = np.array([part.data.shape[0] for part in parts], dtype=INDEX_DTYPE)
    return PerSamplePack(
        indptr=np.concatenate([part.indptr for part in parts]).astype(INDEX_DTYPE, copy=False),
        indices=np.concatenate([part.indices for part in parts]).astype(INDEX_DTYPE, copy=False),
        data=np.concatenate([part.data for part in parts]).astype(VALUE_DTYPE, copy=False),
        sample_offsets=np.concatenate(([0], np.cumsum(nnz_counts))).astype(INDEX_DTYPE),
        shape=shape_array,
    )


def pack_shared_pattern(matrices: Sequence[csr_array]) -> SharedPatternPack:
    """Stack values over one shared pattern into the ``SHARED_PATTERN`` layout.

    Args:
        matrices: CSR samples that already share indptr, indices and shape.
            Callers establish this with ``choose_layout``.

    Returns:
        The single indptr and indices plus ``(N, nnz)`` values.

    Raises:
        ValueError: If the samples do not share a pattern.
    """
    if choose_layout(matrices) is not LayoutType.SHARED_PATTERN:
        raise ValueError("pack_shared_pattern requires all matrices to share one sparsity pattern")
    parts = [split_csr(matrix) for matrix in matrices]
    first = parts[0]
    return SharedPatternPack(
        indptr=first.indptr,
        indices=first.indices,
        data=np.stack([part.data for part in parts]).astype(VALUE_DTYPE, copy=False),
        shape=shape_table([part.shape for part in parts]),
    )
