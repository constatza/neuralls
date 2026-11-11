"""A Matrix Market source file is parsed straight to CSR, never through a dense array."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_array

from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.source_streams import open_matrix_stream
from neuralls.platform.storage.archive_source import FsArchiveSource
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix
from neuralls.shared.types import MatrixFormat

_READERS = MatrixReaders(
    generic=partial(read_matrix, lazy=True), dense=partial(read_dense_npy, lazy=True)
)
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
def mtx_source(tmp_path: Path, spd_dense: np.ndarray) -> Path:
    path = tmp_path / "system.mtx"
    mmwrite(str(path), csr_array(spd_dense))
    return path


def test_mtx_source_loads_csr_without_densifying(
    mtx_source: Path, spd_dense: np.ndarray, toarray_calls: list[int]
) -> None:
    stream = open_matrix_stream(str(mtx_source), readers=_READERS, archive_source=FsArchiveSource())
    loaded = stream.load_sample(0, MatrixFormat.CSR)

    assert isinstance(loaded, csr_array)
    assert toarray_calls == []
    np.testing.assert_array_equal(loaded.toarray(), spd_dense)
