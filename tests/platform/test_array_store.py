"""Offset-write round trips and completeness checks for dense ArrayStore backends."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from tests.conftest import DenseRowsFixture

from neuralls.platform.storage.array_store import ArrayStore, Hdf5ArrayStore, ZarrArrayStore

MATRIX_NAME = "matrix"
VECTOR_NAME = "rhs"
FLOAT64 = "float64"


def _write_in_two_batches(
    store: ArrayStore, fixture: DenseRowsFixture, name: str, data: np.ndarray
) -> None:
    store.write_rows(name, 0, data[: fixture.first_batch_rows])
    store.write_rows(name, fixture.first_batch_rows, data[fixture.first_batch_rows :])


def _read_back(backend: str, path: Path, name: str) -> np.ndarray:
    if backend == "zarr":
        return np.asarray(zarr.open_group(str(path), mode="r")[name][:])
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name][:])


@pytest.mark.parametrize("backend", ["zarr", "hdf5"])
def test_offset_roundtrip(
    backend: str,
    zarr_store_root: Path,
    hdf5_store_path: Path,
    dense_rows_fixture: DenseRowsFixture,
) -> None:
    fx = dense_rows_fixture
    path = zarr_store_root if backend == "zarr" else hdf5_store_path
    store = ZarrArrayStore(path) if backend == "zarr" else Hdf5ArrayStore(path)
    store.create(MATRIX_NAME, (fx.total_rows, *fx.matrices.shape[1:]), FLOAT64)
    store.create(VECTOR_NAME, (fx.total_rows, *fx.vectors.shape[1:]), FLOAT64)
    _write_in_two_batches(store, fx, MATRIX_NAME, fx.matrices)
    _write_in_two_batches(store, fx, VECTOR_NAME, fx.vectors)
    store.close()

    np.testing.assert_array_equal(_read_back(backend, path, MATRIX_NAME), fx.matrices)
    np.testing.assert_array_equal(_read_back(backend, path, VECTOR_NAME), fx.vectors)


@pytest.mark.parametrize("backend", ["zarr", "hdf5"])
def test_short_source_raises_on_close(
    backend: str,
    zarr_store_root: Path,
    hdf5_store_path: Path,
    dense_rows_fixture: DenseRowsFixture,
) -> None:
    fx = dense_rows_fixture
    expected = fx.total_rows
    written = expected - 1
    store = (
        ZarrArrayStore(zarr_store_root) if backend == "zarr" else Hdf5ArrayStore(hdf5_store_path)
    )
    store.create(VECTOR_NAME, (expected, *fx.vectors.shape[1:]), FLOAT64)
    store.write_rows(VECTOR_NAME, 0, fx.vectors[:written])

    with pytest.raises(ValueError):
        store.close()


@pytest.mark.parametrize("backend", ["zarr", "hdf5"])
def test_write_beyond_shape_raises(
    backend: str,
    zarr_store_root: Path,
    hdf5_store_path: Path,
    dense_rows_fixture: DenseRowsFixture,
) -> None:
    fx = dense_rows_fixture
    store = (
        ZarrArrayStore(zarr_store_root) if backend == "zarr" else Hdf5ArrayStore(hdf5_store_path)
    )
    store.create(VECTOR_NAME, (fx.total_rows, *fx.vectors.shape[1:]), FLOAT64)
    with pytest.raises(ValueError):
        store.write_rows(VECTOR_NAME, fx.total_rows - 1, fx.vectors[:2])
    store.write_rows(VECTOR_NAME, 0, fx.vectors)
    store.close()


@pytest.mark.parametrize("backend", ["zarr", "hdf5"])
def test_create_twice_raises(
    backend: str,
    zarr_store_root: Path,
    hdf5_store_path: Path,
    dense_rows_fixture: DenseRowsFixture,
) -> None:
    fx = dense_rows_fixture
    store = (
        ZarrArrayStore(zarr_store_root) if backend == "zarr" else Hdf5ArrayStore(hdf5_store_path)
    )
    store.create(VECTOR_NAME, (fx.total_rows, *fx.vectors.shape[1:]), FLOAT64)
    with pytest.raises(ValueError):
        store.create(VECTOR_NAME, (fx.total_rows, *fx.vectors.shape[1:]), FLOAT64)
    store.write_rows(VECTOR_NAME, 0, fx.vectors)
    store.close()
