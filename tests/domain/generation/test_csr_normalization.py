"""CSR-aware matrix normalization and sparse source loading for generation."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_array

from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.helpers import normalize_matrix_for_generation
from neuralls.domain.generation.source_streams import open_matrix_stream
from neuralls.domain.normalization import matrix_norm
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix
from neuralls.shared.types import MatrixFormat, MatrixNormType

_READER = partial(read_matrix, lazy=True)
_READERS = MatrixReaders(generic=_READER, dense=partial(read_dense_npy, lazy=True))

_SEED = 20260610
_SIZE = 40
_DENSITY_THRESHOLD = 0.6


@pytest.fixture
def spd_dense() -> np.ndarray:
    """Seeded sparse-pattern symmetric positive-definite matrix, dense storage."""
    rng = np.random.default_rng(_SEED)
    raw = rng.standard_normal((_SIZE, _SIZE))
    sym = 0.5 * (raw + raw.T)
    sym[np.abs(sym) < _DENSITY_THRESHOLD] = 0.0
    return sym + _SIZE * np.eye(_SIZE)


@pytest.fixture
def spd_csr(spd_dense: np.ndarray) -> csr_array:
    return csr_array(spd_dense)


@pytest.fixture
def toarray_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count every densification request made through scipy's sparse API."""
    calls: list[int] = []
    original = csr_array.toarray

    def counting_toarray(self: csr_array) -> np.ndarray:
        calls.append(1)
        return original(self)

    monkeypatch.setattr(csr_array, "toarray", counting_toarray)
    return calls


@pytest.fixture
def mtx_source(tmp_path: Path, spd_csr: csr_array) -> Path:
    path = tmp_path / "system.mtx"
    mmwrite(str(path), spd_csr)
    return path


@pytest.fixture
def npy_source(tmp_path: Path, spd_dense: np.ndarray) -> Path:
    path = tmp_path / "system.npy"
    np.save(path, spd_dense)
    return path


@pytest.mark.parametrize("normalize_type", ["matrix", "none"])
def test_csr_normalization_matches_dense(
    spd_dense: np.ndarray, spd_csr: csr_array, normalize_type: str
) -> None:
    dense_norm, dense_scale, dense_value_scale = normalize_matrix_for_generation(
        spd_dense, normalize_type, spectral_radius_bound=None
    )
    csr_norm, csr_scale, csr_value_scale = normalize_matrix_for_generation(
        spd_csr, normalize_type, spectral_radius_bound=None
    )

    assert isinstance(csr_norm, csr_array)
    assert isinstance(dense_norm, np.ndarray)
    np.testing.assert_allclose(csr_norm.toarray(), dense_norm, rtol=1e-12, atol=0.0)
    assert csr_value_scale == pytest.approx(dense_value_scale, rel=1e-12)
    assert (csr_scale is None) == (dense_scale is None)


@pytest.mark.parametrize("kind", [MatrixNormType.ONE, MatrixNormType.INF, MatrixNormType.FROBENIUS])
def test_csr_norm_matches_dense(
    spd_dense: np.ndarray, spd_csr: csr_array, kind: MatrixNormType
) -> None:
    assert matrix_norm(spd_csr, kind) == pytest.approx(matrix_norm(spd_dense, kind), rel=1e-12)


def test_csr_spectral_norm_matches_dense(spd_dense: np.ndarray, spd_csr: csr_array) -> None:
    # svds is iterative, so the spectral norm agrees to solver tolerance, not 1e-12.
    assert matrix_norm(spd_csr, MatrixNormType.SPECTRAL) == pytest.approx(
        matrix_norm(spd_dense, MatrixNormType.SPECTRAL), rel=1e-8
    )


def test_csr_normalization_never_densifies(spd_csr: csr_array, toarray_calls: list[int]) -> None:
    normalized, _, _ = normalize_matrix_for_generation(spd_csr, "matrix", None)
    matrix_norm(normalized, MatrixNormType.INF)
    assert isinstance(normalized, csr_array)
    assert toarray_calls == []


def test_mtx_source_loads_csr_without_densifying(
    mtx_source: Path, spd_dense: np.ndarray, toarray_calls: list[int]
) -> None:
    stream = open_matrix_stream(str(mtx_source), readers=_READERS)
    loaded = stream.load_sample(0, MatrixFormat.CSR)

    assert isinstance(loaded, csr_array)
    assert toarray_calls == []
    np.testing.assert_array_equal(loaded.toarray(), spd_dense)


def test_dense_npy_source_requested_as_csr_converts_explicitly(
    npy_source: Path, spd_dense: np.ndarray
) -> None:
    stream = open_matrix_stream(str(npy_source), readers=_READERS)
    loaded = stream.load_sample(0, MatrixFormat.CSR)

    assert isinstance(loaded, csr_array)
    np.testing.assert_array_equal(loaded.toarray(), spd_dense)


def test_dense_format_returns_ndarray(mtx_source: Path, spd_dense: np.ndarray) -> None:
    stream = open_matrix_stream(str(mtx_source), readers=_READERS)
    loaded = stream.load_sample(0, MatrixFormat.DENSE)

    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, spd_dense)
