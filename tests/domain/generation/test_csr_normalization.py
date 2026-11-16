"""CSR-aware matrix scaling and sparse source loading for generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.generation.source_streams import open_matrix_stream
from neuralls.domain.normalization import matrix_norm, scale_factor, scale_matrix
from neuralls.shared.system_scaling import GershgorinScaling, NoScaling, SystemScaling
from neuralls.shared.types import MatrixFormat, MatrixNormType
from tests.fakes.archive_source import InMemoryArchiveSource

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
def mtx_source(archive_source: InMemoryArchiveSource, spd_csr: csr_array) -> Path:
    """The sparse system registered under a ``.mtx`` path."""
    return archive_source.add(Path("archive/system.mtx"), spd_csr)


@pytest.fixture
def npy_source(archive_source: InMemoryArchiveSource, spd_dense: np.ndarray) -> Path:
    """The dense system registered under a ``.npy`` path."""
    return archive_source.add(Path("archive/system.npy"), spd_dense)


@pytest.mark.parametrize("scaling", [GershgorinScaling(), NoScaling()], ids=["gershgorin", "none"])
def test_csr_scaling_matches_dense(
    spd_dense: np.ndarray, spd_csr: csr_array, scaling: SystemScaling
) -> None:
    dense_alpha = scale_factor(spd_dense, scaling)
    csr_alpha = scale_factor(spd_csr, scaling)
    dense_scaled = scale_matrix(spd_dense, dense_alpha)
    csr_scaled = scale_matrix(spd_csr, csr_alpha)

    assert isinstance(csr_scaled, csr_array)
    assert isinstance(dense_scaled, np.ndarray)
    np.testing.assert_allclose(csr_scaled.toarray(), dense_scaled, rtol=1e-12, atol=0.0)
    assert csr_alpha == pytest.approx(dense_alpha, rel=1e-12)


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


def test_csr_scaling_never_densifies(spd_csr: csr_array, toarray_calls: list[int]) -> None:
    scaled = scale_matrix(spd_csr, scale_factor(spd_csr, GershgorinScaling()))
    matrix_norm(scaled, MatrixNormType.INF)
    assert isinstance(scaled, csr_array)
    assert toarray_calls == []


def test_dense_npy_source_requested_as_csr_converts_explicitly(
    npy_source: Path, spd_dense: np.ndarray, archive_source: InMemoryArchiveSource
) -> None:
    stream = open_matrix_stream(
        str(npy_source), readers=archive_source.readers(), archive_source=archive_source
    )
    loaded = stream.load_sample(0, MatrixFormat.CSR)

    assert isinstance(loaded, csr_array)
    np.testing.assert_array_equal(loaded.toarray(), spd_dense)


def test_dense_format_returns_ndarray(
    mtx_source: Path, spd_dense: np.ndarray, archive_source: InMemoryArchiveSource
) -> None:
    stream = open_matrix_stream(
        str(mtx_source), readers=archive_source.readers(), archive_source=archive_source
    )
    loaded = stream.load_sample(0, MatrixFormat.DENSE)

    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, spd_dense)
