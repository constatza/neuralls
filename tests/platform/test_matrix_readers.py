"""Tests for the suffix-keyed matrix reader registry."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_array, save_npz

from neuralls.platform.storage.matrix_readers import (
    MTX_GZ_SUFFIX,
    MTX_SUFFIX,
    NPY_SUFFIX,
    NPZ_SUFFIX,
    TXT_SUFFIX,
    read_matrix,
    to_dense,
)

SEED = 0
MATRIX_SIZE = 6
SYMMETRIC_STORED_LOWER_HEADER = "%%MatrixMarket matrix coordinate real symmetric\n"


@pytest.fixture
def dense_reference() -> np.ndarray:
    """Seeded sparse-pattern dense matrix used for round-trip comparisons."""
    rng = np.random.default_rng(SEED)
    values = rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))
    mask = rng.random((MATRIX_SIZE, MATRIX_SIZE)) < 0.4
    return np.where(mask, values, 0.0)


@pytest.fixture
def symmetric_reference(dense_reference: np.ndarray) -> np.ndarray:
    """Full symmetric dense matrix derived from the seeded reference."""
    return dense_reference + dense_reference.T


def _write_single_triangle_mtx(path: Path, full: np.ndarray) -> None:
    """Write only the lower triangle of ``full`` under a MatrixMarket symmetric header."""
    rows, cols = np.nonzero(np.tril(full))
    lines = [SYMMETRIC_STORED_LOWER_HEADER, f"{full.shape[0]} {full.shape[1]} {rows.size}\n"]
    lines.extend(
        f"{r + 1} {c + 1} {float(full[r, c])!r}\n" for r, c in zip(rows, cols, strict=True)
    )
    path.write_text("".join(lines), encoding="utf-8")


def _write_matrix(path: Path, matrix: np.ndarray, suffix: str) -> None:
    """Persist ``matrix`` in the on-disk format identified by ``suffix``."""
    match suffix:
        case ".npy":
            np.save(path, matrix)
        case ".txt":
            np.savetxt(path, matrix)
        case ".npz":
            save_npz(path, csr_array(matrix))
        case ".mtx":
            mmwrite(path, csr_array(matrix))
        case ".mtx.gz":
            with gzip.open(path, "wb") as stream:
                mmwrite(stream, csr_array(matrix))
        case _:
            raise AssertionError(f"no writer for suffix {suffix}")


@pytest.mark.parametrize(
    ("suffix", "expected_type"),
    [
        (NPY_SUFFIX, np.ndarray),
        (TXT_SUFFIX, np.ndarray),
        (NPZ_SUFFIX, csr_array),
        (MTX_SUFFIX, csr_array),
        (MTX_GZ_SUFFIX, csr_array),
    ],
)
def test_round_trip_per_suffix(
    tmp_path: Path,
    dense_reference: np.ndarray,
    suffix: str,
    expected_type: type,
) -> None:
    path = tmp_path / f"matrix{suffix}"
    _write_matrix(path, dense_reference, suffix)

    loaded = read_matrix(path)

    assert isinstance(loaded, expected_type)
    np.testing.assert_allclose(to_dense(loaded), dense_reference)


def test_unknown_suffix_raises_with_supported_list(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="supported suffixes"):
        read_matrix(path)


def test_npz_with_dense_array_raises(tmp_path: Path, dense_reference: np.ndarray) -> None:
    path = tmp_path / "dense.npz"
    np.savez(path, matrix=dense_reference)

    with pytest.raises(ValueError, match="dense.npz"):
        read_matrix(path)


def test_symmetric_mtx_written_by_mmwrite_expands_to_full(
    tmp_path: Path, symmetric_reference: np.ndarray
) -> None:
    path = tmp_path / "symmetric_auto.mtx"
    mmwrite(path, csr_array(symmetric_reference))
    assert "symmetric" in path.read_text(encoding="utf-8").splitlines()[0]

    loaded = read_matrix(path)

    assert isinstance(loaded, csr_array)
    np.testing.assert_allclose(loaded.toarray(), symmetric_reference)


def test_symmetric_single_triangle_mtx_expands_to_full(
    tmp_path: Path, symmetric_reference: np.ndarray
) -> None:
    path = tmp_path / "symmetric_lower.mtx"
    _write_single_triangle_mtx(path, symmetric_reference)

    loaded = read_matrix(path)

    assert isinstance(loaded, csr_array)
    np.testing.assert_allclose(loaded.toarray(), symmetric_reference)
