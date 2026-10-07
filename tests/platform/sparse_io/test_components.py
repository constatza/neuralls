"""Tests for the pure CSR component conversions."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import CsrComponents, to_components, to_scipy
from neuralls.platform.sparse_io.torch_convert import to_torch_csr


def test_components_round_trip_preserves_empty_rows_and_cols(
    csr_with_empty_rows_and_cols: csr_array,
) -> None:
    restored = to_scipy(to_components(csr_with_empty_rows_and_cols))

    assert restored.shape == (4, 6)
    np.testing.assert_array_equal(restored.toarray(), csr_with_empty_rows_and_cols.toarray())


def test_to_torch_csr_matches_dense(csr_with_empty_rows_and_cols: csr_array) -> None:
    tensor = to_torch_csr(to_components(csr_with_empty_rows_and_cols))

    assert tensor.layout == torch.sparse_csr
    assert tensor.dtype == torch.float64
    np.testing.assert_array_equal(tensor.to_dense().numpy(), csr_with_empty_rows_and_cols.toarray())


def test_components_have_integer_structure_and_float_values(
    csr_with_empty_rows_and_cols: csr_array,
) -> None:
    components = to_components(csr_with_empty_rows_and_cols)

    assert isinstance(components, CsrComponents)
    assert components.indptr.dtype == np.int64
    assert components.indices.dtype == np.int64
    assert components.data.dtype == np.float64


def test_explicit_zero_is_kept_as_stored_entry(csr_with_stored_zero: csr_array) -> None:
    components = to_components(csr_with_stored_zero)

    assert components.data.shape[0] == 3
    assert components.indices.tolist() == [0, 1, 1]
    restored = to_scipy(components)
    assert restored.nnz == 3
