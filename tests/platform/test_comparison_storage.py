"""Tests for comparison dataset I/O helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_array

from neuralls.platform.storage.comparison import (
    COMPARISON_MATRIX_FORMAT,
    load_comparison_matrix,
    load_system_extras,
)
from neuralls.platform.storage.generation_formats import HDF5_FILENAME
from neuralls.shared.types import MatrixFormat

_REPRESENTATION: dict[MatrixFormat, type] = {
    MatrixFormat.CSR: csr_array,
    MatrixFormat.DENSE: np.ndarray,
}
"""In-memory type of each matrix format."""


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Dataset directory with coordinates.zarr/ (3 samples, 2×3) and params.npy."""
    coords = np.arange(18.0, dtype=np.float64).reshape(3, 2, 3)
    zarr.save(str(tmp_path / "coordinates.zarr"), coords)
    np.save(tmp_path / "params.npy", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    return tmp_path


@pytest.fixture
def hdf5_dataset_dir(tmp_path: Path) -> Path:
    """Dataset directory with a single dataset.h5 holding a named "weights" array."""
    with h5py.File(str(tmp_path / HDF5_FILENAME), "w") as f:
        f.create_dataset("weights", data=np.arange(15.0, dtype=np.float64).reshape(3, 5))
    return tmp_path


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


def test_load_zarr_extra_by_name(dataset_dir: Path) -> None:
    """Load a zarr array extra by name, extracting specified sample."""
    result = load_system_extras(dataset_dir, frozenset({"coordinates"}), sample_index=1)
    assert "coordinates" in result
    assert result["coordinates"].shape == (2, 3)
    np.testing.assert_allclose(result["coordinates"], np.arange(6.0, 12.0).reshape(2, 3))


def test_load_npy_extra_by_name(dataset_dir: Path) -> None:
    """Load a npy file extra by name."""
    result = load_system_extras(dataset_dir, frozenset({"params"}), sample_index=0)
    assert "params" in result
    np.testing.assert_allclose(result["params"], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_load_hdf5_extra_by_name(hdf5_dataset_dir: Path) -> None:
    """Load a named dataset out of a single-file dataset.h5 extra."""
    result = load_system_extras(hdf5_dataset_dir, frozenset({"weights"}), sample_index=1)
    assert "weights" in result
    np.testing.assert_allclose(result["weights"], np.arange(5.0, 10.0))


def test_missing_extra_silently_absent(dataset_dir: Path) -> None:
    """Missing extras are not included in the result dict."""
    result = load_system_extras(dataset_dir, frozenset({"nonexistent"}), sample_index=0)
    assert "nonexistent" not in result


def test_empty_names_returns_empty_dict(dataset_dir: Path) -> None:
    """Empty names set returns empty dict."""
    result = load_system_extras(dataset_dir, frozenset(), sample_index=0)
    assert result == {}
