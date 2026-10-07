"""Plain CSR component container and its conversions to and from ``csr_array``.

The components are the canonical in-memory form shared by every storage
backend and by the torch bridge. Conversions perform no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

INDEX_DTYPE: Final = np.int64
VALUE_DTYPE: Final = np.float64


@dataclass(frozen=True)
class CsrComponents:
    """Compressed sparse row arrays plus the ``(rows, cols)`` shape of one matrix."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    shape: tuple[int, int]


def to_components(matrix: csr_array) -> CsrComponents:
    """Split a ``csr_array`` into canonical components.

    Args:
        matrix: CSR matrix. It is copied and its indices sorted, so the caller's
            matrix is not modified. Explicitly stored zeros are kept.

    Returns:
        The components with the integer and float dtypes of the storage schema.
    """
    canonical = csr_array(matrix, copy=True)
    canonical.sort_indices()
    rows, cols = canonical.shape
    return CsrComponents(
        indptr=canonical.indptr.astype(INDEX_DTYPE, copy=False),
        indices=canonical.indices.astype(INDEX_DTYPE, copy=False),
        data=canonical.data.astype(VALUE_DTYPE, copy=False),
        shape=(int(rows), int(cols)),
    )


def to_scipy(components: CsrComponents) -> csr_array:
    """Assemble a ``csr_array`` from components.

    Args:
        components: Row pointer, column indices, values and ``(rows, cols)`` shape.

    Returns:
        The CSR matrix, using the component arrays without forcing a copy.

    Raises:
        ValueError: If the lengths are inconsistent or ``indptr[-1] != len(data)``.
    """
    rows = components.shape[0]
    if components.indptr.shape != (rows + 1,):
        raise ValueError(
            f"indptr must have length {rows + 1} for {rows} rows, got {components.indptr.shape}"
        )
    if components.indices.shape != components.data.shape:
        raise ValueError(
            "indices and data must have equal length, got "
            f"{components.indices.shape[0]} and {components.data.shape[0]}"
        )
    stored = int(components.indptr[-1])
    if stored != components.data.shape[0]:
        raise ValueError(
            f"indptr[-1]={stored} must equal the number of stored values {components.data.shape[0]}"
        )
    return csr_array(
        (components.data, components.indices, components.indptr), shape=components.shape
    )
