"""Tests for comparison dataset I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.platform.storage.comparison import (
    COMPARISON_MATRIX_FORMAT,
    load_comparison_matrix,
)
from neuralls.shared.types import MatrixFormat

_REPRESENTATION: dict[MatrixFormat, type] = {
    MatrixFormat.CSR: csr_array,
    MatrixFormat.DENSE: np.ndarray,
}
"""In-memory type of each matrix format."""


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    """A raw 2x2 matrix stored densely in a .npy file."""
    path = tmp_path / "matrix.npy"
    np.save(path, np.array([[2.0, 0.0], [0.0, 3.0]]))
    return path


def test_comparison_matrix_loads_in_the_format_its_identity_declares(matrix_file: Path) -> None:
    """The format hashed into comparison identities is the one the loader returns."""
    matrix = load_comparison_matrix(matrix_file)

    assert isinstance(matrix, _REPRESENTATION[COMPARISON_MATRIX_FORMAT])
    np.testing.assert_array_equal(matrix.toarray(), [[2.0, 0.0], [0.0, 3.0]])
