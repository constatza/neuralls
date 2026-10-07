"""Graph view of a sparse system matrix (pure, no I/O).

A CSR matrix is already an adjacency structure: each stored entry is a directed
edge from its row (source) to its column (target), weighted by its value. The
view is computed on demand from the stored matrix and never persisted, so the
dataset keeps a single copy of the matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_array


def csr_to_edge_index(matrix: csr_array) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the matrix as a COO-style edge list for graph learning.

    Every stored entry becomes one directed edge, including explicitly stored
    zeros, so the output round-trips through ``csr_array((edge_attr, edge_index),
    shape=...)`` without losing structure.

    Args:
        matrix: Square sparse matrix in CSR form.

    Returns:
        ``edge_index`` of shape ``(2, nnz)`` with int64 ``(row, col)`` rows, and
        ``edge_attr`` of shape ``(nnz,)`` with float64 values in the same order.

    Raises:
        ValueError: If ``matrix`` is not square.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Edge view requires a square matrix, got shape {matrix.shape}")
    coo = matrix.tocoo()
    edge_index = torch.as_tensor(np.stack([coo.row, coo.col]).astype(np.int64))
    edge_attr = torch.as_tensor(coo.data.astype(np.float64))
    return edge_index, edge_attr
