"""Tests for the pure CSR-to-edge-list view in ``neuralls.domain.graph``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array

from neuralls.domain.graph import csr_to_edge_index

SEED = 20260610
SPD_SIZE = 12
ZERO_FRACTION = 0.3


@pytest.fixture
def spd_like_dense() -> np.ndarray:
    """Seeded symmetric dense matrix with a dominant diagonal and ~30% zeros."""
    rng = np.random.default_rng(SEED)
    dense = rng.standard_normal((SPD_SIZE, SPD_SIZE))
    dense[rng.random((SPD_SIZE, SPD_SIZE)) < ZERO_FRACTION] = 0.0
    sym = (dense + dense.T) / 2.0
    sym += SPD_SIZE * np.eye(SPD_SIZE)
    return sym


@pytest.fixture
def csr_fixture(spd_like_dense: np.ndarray) -> csr_array:
    """Sparse CSR form of the seeded SPD-like matrix (no explicit zeros)."""
    return csr_array(spd_like_dense)


@pytest.fixture
def explicit_zero_csr() -> csr_array:
    """2x2 CSR matrix that stores 0.0 explicitly at position (0, 1)."""
    data = np.array([1.0, 0.0, 2.0])
    indices = np.array([0, 1, 1])
    indptr = np.array([0, 2, 3])
    return csr_array((data, indices, indptr), shape=(2, 2))


@pytest.fixture
def rectangular_csr() -> csr_array:
    """Non-square 2x3 CSR matrix."""
    rng = np.random.default_rng(SEED)
    return csr_array(rng.standard_normal((2, 3)))


@pytest.fixture
def single_entry_csr() -> csr_array:
    """1x1 CSR matrix with one stored entry."""
    return csr_array(np.array([[3.5]]))


def _rebuild(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, shape: tuple[int, int]
) -> csr_array:
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    return csr_array((edge_attr.numpy(), (row, col)), shape=shape)


def test_round_trip_reproduces_matrix_exactly(csr_fixture: csr_array) -> None:
    edge_index, edge_attr = csr_to_edge_index(csr_fixture)

    rebuilt = _rebuild(edge_index, edge_attr, csr_fixture.shape)

    difference = rebuilt - csr_fixture
    assert difference.nnz == 0
    np.testing.assert_array_equal(rebuilt.toarray(), csr_fixture.toarray())


def test_explicit_stored_zero_is_kept_as_edge(explicit_zero_csr: csr_array) -> None:
    edge_index, edge_attr = csr_to_edge_index(explicit_zero_csr)

    assert edge_index.shape[1] == explicit_zero_csr.nnz == 3
    zero_positions = (edge_attr == 0.0).nonzero(as_tuple=True)[0]
    assert zero_positions.numel() == 1
    idx = int(zero_positions[0])
    assert (int(edge_index[0, idx]), int(edge_index[1, idx])) == (0, 1)

    rebuilt = _rebuild(edge_index, edge_attr, explicit_zero_csr.shape)
    assert rebuilt.nnz == explicit_zero_csr.nnz


def test_edges_follow_csr_row_major_order(csr_fixture: csr_array) -> None:
    edge_index, edge_attr = csr_to_edge_index(csr_fixture)

    rows = edge_index[0]
    assert torch.all(rows[1:] >= rows[:-1]).item()
    np.testing.assert_array_equal(edge_attr.numpy(), csr_fixture.tocoo().data)


def test_non_square_matrix_raises_value_error(rectangular_csr: csr_array) -> None:
    with pytest.raises(ValueError, match="square"):
        csr_to_edge_index(rectangular_csr)


def test_single_entry_matrix_gives_one_edge_at_origin(single_entry_csr: csr_array) -> None:
    edge_index, edge_attr = csr_to_edge_index(single_entry_csr)

    assert edge_index.shape == (2, 1)
    assert int(edge_index[0, 0]) == 0
    assert int(edge_index[1, 0]) == 0
    assert float(edge_attr[0]) == pytest.approx(3.5)
