"""Sparsity-pattern identity of CSR samples for the shared-pattern layout.

A pattern is a matrix's shape plus its ``indptr`` and ``indices`` after canonicalisation.
Stored values never take part: two samples with the same stored positions share a pattern
even when their values differ. Stored explicit zeros are stored entries, so they count.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import INDEX_DTYPE

__all__ = ["canonicalise_csr", "csr_pattern_matches"]


def canonicalise_csr(matrix: csr_array) -> csr_array:
    """Return a copy with indices sorted within each row and duplicate entries summed.

    Canonical form makes two matrices with the same stored positions compare equal, whatever
    order their indices were written in. Explicitly stored zeros are kept: the shared layout
    stores them, so dropping them would change the pattern that is written.

    Args:
        matrix: CSR matrix. It is not modified.

    Returns:
        A new canonical ``csr_array``.
    """
    canonical = csr_array(matrix, copy=True)
    canonical.sum_duplicates()
    return canonical


def csr_pattern_matches(reference: csr_array, sample: csr_array) -> bool:
    """Whether two canonical CSR matrices share a sparsity pattern.

    Compares shape first (constant time), then ``indptr`` and ``indices`` in one pass each.
    Values are never compared. The inputs must already be canonical (see
    ``canonicalise_csr``); the check does not densify either matrix.

    Args:
        reference: Canonical matrix whose pattern is the reference (the first sample).
        sample: Canonical matrix to test against the reference.

    Returns:
        True when shape, ``indptr`` and ``indices`` are all equal.
    """
    return (
        reference.shape == sample.shape
        and np.array_equal(
            reference.indptr.astype(INDEX_DTYPE, copy=False),
            sample.indptr.astype(INDEX_DTYPE, copy=False),
        )
        and np.array_equal(
            reference.indices.astype(INDEX_DTYPE, copy=False),
            sample.indices.astype(INDEX_DTYPE, copy=False),
        )
    )
