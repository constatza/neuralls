"""Pure layout selection and packing for CSR samples stored as flat arrays.

The on-disk schema, member by member, is documented in the storage package
(``neuralls.platform.storage.csr_layout``). This module only decides which
layout applies and builds the flat arrays for it; it performs no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import (
    INDEX_DTYPE,
    VALUE_DTYPE,
    to_components,
)
from neuralls.shared.types import LayoutType

INDPTR_ARRAY: Final = "indptr"
INDICES_ARRAY: Final = "indices"
DATA_ARRAY: Final = "data"
SAMPLE_OFFSETS_ARRAY: Final = "sample_offsets"
SHAPE_ARRAY: Final = "shape"
"""Member name of the (N, 2) int64 per-sample shape array."""

SHAPE_COLUMNS: Final = 2
"""A matrix shape is (rows, cols)."""


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
    first = to_components(matrices[0])
    for matrix in matrices[1:]:
        other = to_components(matrix)
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
        The flat arrays and offsets described in the storage module docstring.
    """
    parts = [to_components(matrix) for matrix in matrices]
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
    parts = [to_components(matrix) for matrix in matrices]
    first = parts[0]
    return SharedPatternPack(
        indptr=first.indptr,
        indices=first.indices,
        data=np.stack([part.data for part in parts]).astype(VALUE_DTYPE, copy=False),
        shape=shape_table([part.shape for part in parts]),
    )


def members_for(
    layout: LayoutType, matrices: Sequence[csr_array]
) -> Mapping[str, NDArray[np.generic]]:
    """Map every member name of the layout to its flat array. Pure, no I/O.

    Args:
        layout: Layout chosen for ``matrices`` by ``choose_layout``.
        matrices: CSR samples packed under ``layout``.

    Returns:
        Member name to array, with the names of the module-level constants.
    """
    if layout is LayoutType.MANY_MATRICES:
        per_sample = pack_per_sample(matrices)
        return {
            INDPTR_ARRAY: per_sample.indptr,
            INDICES_ARRAY: per_sample.indices,
            DATA_ARRAY: per_sample.data,
            SAMPLE_OFFSETS_ARRAY: per_sample.sample_offsets,
            SHAPE_ARRAY: per_sample.shape,
        }
    shared = pack_shared_pattern(matrices)
    return {
        INDPTR_ARRAY: shared.indptr,
        INDICES_ARRAY: shared.indices,
        DATA_ARRAY: shared.data,
        SHAPE_ARRAY: shared.shape,
    }
