"""Directory-per-sample raw layouts: ``<root>/<id>/K_ff.mtx``.

The new raw format lays out one subdirectory per sample, each holding a
constant-named MatrixMarket file (``K_ff.mtx``). Unlike today's flat
filename-glob layout, the sample id lives in the directory name, not the
file name, and the glob's wildcard spans a whole path segment rather than
part of a filename. These tests pin: multi-depth glob resolution
(``*`` and ``**``), directory-derived sample ids, and that the existing
filename-glob behavior (``A_*.txt``) is untouched.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_array

from neuralls.domain.generation.source_streams import GlobMatrixStream
from neuralls.platform.storage.matrix_readers import read_matrix
from neuralls.shared.types import MatrixFormat

_READER = partial(read_matrix, lazy=True)


def _marker_matrix(value: float) -> csr_array:
    """A 2x2 sparse matrix whose values identify one source file."""
    return csr_array(np.array([[value, 0.0], [0.0, value + 1.0]], dtype=np.float64))


@pytest.fixture
def one_level_case_dirs(tmp_path: Path) -> Path:
    """<root>/{0,1,2}/K_ff.mtx, each a distinct sparse matrix."""
    root = tmp_path / "raw"
    root.mkdir()
    for i in (0, 1, 2):
        case_dir = root / str(i)
        case_dir.mkdir()
        mmwrite(str(case_dir / "K_ff.mtx"), _marker_matrix(float(i)))
    return root


@pytest.fixture
def nested_case_dirs(tmp_path: Path) -> Path:
    """<root>/case_a/{0,1}/K_ff.mtx, one level deeper than one_level_case_dirs."""
    root = tmp_path / "raw_nested"
    root.mkdir()
    case_a = root / "case_a"
    case_a.mkdir()
    for i in (0, 1):
        case_dir = case_a / str(i)
        case_dir.mkdir()
        mmwrite(str(case_dir / "K_ff.mtx"), _marker_matrix(float(i)))
    return root


def test_one_level_wildcard_derives_ids_from_directory_name(
    one_level_case_dirs: Path,
) -> None:
    stream = GlobMatrixStream(str(one_level_case_dirs / "*" / "K_ff.mtx"), reader=_READER)

    assert stream.sample_ids == (0, 1, 2)
    sample = stream.load_sample(1, matrix_format=MatrixFormat.CSR)
    np.testing.assert_array_equal(csr_array(sample).toarray(), _marker_matrix(1.0).toarray())


def test_recursive_wildcard_resolves_nested_directories(nested_case_dirs: Path) -> None:
    stream = GlobMatrixStream(str(nested_case_dirs / "**" / "K_ff.mtx"), reader=_READER)

    assert stream.sample_ids == (0, 1)


def test_one_level_wildcard_include_indices_keeps_directory_derived_ids(
    one_level_case_dirs: Path,
) -> None:
    stream = GlobMatrixStream(
        str(one_level_case_dirs / "*" / "K_ff.mtx"), include_indices=(0, 2), reader=_READER
    )

    assert stream.sample_ids == (0, 2)


def test_one_level_wildcard_custom_regex_on_directory_name(tmp_path: Path) -> None:
    """A non-numeric-only directory name (``case_3``) still yields a usable id."""
    root = tmp_path / "raw_prefixed"
    root.mkdir()
    for i in (3, 7):
        case_dir = root / f"case_{i}"
        case_dir.mkdir()
        mmwrite(str(case_dir / "K_ff.mtx"), _marker_matrix(float(i)))

    stream = GlobMatrixStream(
        str(root / "*" / "K_ff.mtx"), sample_id_regex=r"case_(\d+)", reader=_READER
    )

    assert stream.sample_ids == (3, 7)


def test_flat_filename_glob_is_unaffected_by_directory_id_support(tmp_path: Path) -> None:
    """Regression: a flat A_*.txt layout still derives ids from the filename, not a directory."""
    mat_dir = tmp_path / "flat"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "A_5.txt", np.eye(2))
    np.savetxt(mat_dir / "A_9.txt", 2 * np.eye(2))

    stream = GlobMatrixStream(str(mat_dir / "A_*.txt"), reader=_READER)

    assert stream.sample_ids == (5, 9)
