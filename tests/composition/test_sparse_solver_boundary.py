"""Representation tests for the dense/CSR solver boundary."""

from __future__ import annotations

import pytest
import torch
from scipy.sparse import csr_array, diags, identity, kron

from neuralls.composition.solvers.torchalg_runner import _to_torch_operator

GRID_SIDE = 8
"""Side of the 2D Laplacian; the system has GRID_SIDE**2 = 64 unknowns."""


@pytest.fixture
def laplacian_csr() -> csr_array:
    """Symmetric positive-definite 2D 5-point Laplacian as CSR."""
    one_d = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(GRID_SIDE, GRID_SIDE))
    eye = identity(GRID_SIDE, format="csr")
    return csr_array(kron(eye, one_d) + kron(one_d, eye))


def test_to_torch_operator_builds_csr_layout_for_csr_input(laplacian_csr: csr_array) -> None:
    """A CSR input becomes a sparse CSR tensor with the same nnz and shape."""
    operator = _to_torch_operator(laplacian_csr)

    assert operator.layout == torch.sparse_csr
    assert operator.dtype == torch.float64
    assert tuple(operator.shape) == laplacian_csr.shape
    assert operator._nnz() == laplacian_csr.nnz


def test_to_torch_operator_keeps_dense_input_dense(laplacian_csr: csr_array) -> None:
    """A dense input stays a dense float64 tensor."""
    operator = _to_torch_operator(laplacian_csr.toarray())

    assert operator.layout == torch.strided
    assert operator.dtype == torch.float64
