"""On-disk layout of CSR system matrices stored as a zarr or hdf5 group.

The storage schema is documented here. Each stored matrix is one group (a zarr
group ``matrix/``, or the hdf5 group ``matrix`` in ``dataset.h5``) holding these
members. The backends that read and write them live in ``neuralls.platform.sparse_io``:

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

The pure conversions and layout packing live in ``neuralls.platform.sparse_io``.
This module re-exports them under their storage names and keeps the array-level
``build_csr`` entry point that readers use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import (
    INDEX_DTYPE,
    VALUE_DTYPE,
    CsrComponents,
    to_scipy,
)
from neuralls.platform.sparse_io.components import (
    to_components as split_csr,
)
from neuralls.platform.sparse_io.layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
    SHAPE_COLUMNS,
    PerSamplePack,
    SharedPatternPack,
    choose_layout,
    indptr_offsets,
    pack_per_sample,
    pack_shared_pattern,
    shape_table,
)

__all__ = [
    "DATA_ARRAY",
    "INDEX_DTYPE",
    "INDICES_ARRAY",
    "INDPTR_ARRAY",
    "SAMPLE_OFFSETS_ARRAY",
    "SHAPE_ARRAY",
    "SHAPE_COLUMNS",
    "VALUE_DTYPE",
    "PerSamplePack",
    "SharedPatternPack",
    "build_csr",
    "choose_layout",
    "indptr_offsets",
    "pack_per_sample",
    "pack_shared_pattern",
    "shape_table",
    "split_csr",
]


def build_csr(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    shape: tuple[int, int],
) -> csr_array:
    """Assemble a ``csr_array`` from its three arrays and shape.

    Raises:
        ValueError: If the lengths are inconsistent or ``indptr[-1] != len(data)``.
    """
    return to_scipy(CsrComponents(indptr=indptr, indices=indices, data=data, shape=shape))
