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

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.generation.source_streams import GlobMatrixStream
from neuralls.shared.types import MatrixFormat
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.os_paths import FAKE_ROOT

RAW_ROOT = FAKE_ROOT / "archive/raw"


def _marker_matrix(value: float) -> csr_array:
    """A 2x2 sparse matrix whose values identify one source file."""
    return csr_array(np.array([[value, 0.0], [0.0, value + 1.0]], dtype=np.float64))


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def one_level_case_dirs(archive_source: InMemoryArchiveSource) -> Path:
    """<root>/{0,1,2}/K_ff.mtx, each a distinct sparse matrix."""
    for i in (0, 1, 2):
        archive_source.add(RAW_ROOT / str(i) / "K_ff.mtx", _marker_matrix(float(i)))
    return RAW_ROOT


@pytest.fixture
def nested_case_dirs(archive_source: InMemoryArchiveSource) -> Path:
    """<root>/case_a/{0,1}/K_ff.mtx, one level deeper than one_level_case_dirs."""
    for i in (0, 1):
        archive_source.add(RAW_ROOT / "case_a" / str(i) / "K_ff.mtx", _marker_matrix(float(i)))
    return RAW_ROOT


@pytest.fixture
def prefixed_case_dirs(archive_source: InMemoryArchiveSource) -> Path:
    """<root>/case_{3,7}/K_ff.mtx: directory names that are not bare integers."""
    for i in (3, 7):
        archive_source.add(RAW_ROOT / f"case_{i}" / "K_ff.mtx", _marker_matrix(float(i)))
    return RAW_ROOT


@pytest.fixture
def flat_filename_dir(archive_source: InMemoryArchiveSource) -> Path:
    """<root>/A_5.txt and A_9.txt: the flat filename-glob layout."""
    archive_source.add(RAW_ROOT / "A_5.txt", np.eye(2))
    archive_source.add(RAW_ROOT / "A_9.txt", 2 * np.eye(2))
    return RAW_ROOT


def test_one_level_wildcard_derives_ids_from_directory_name(
    one_level_case_dirs: Path, archive_source: InMemoryArchiveSource
) -> None:
    stream = GlobMatrixStream(
        str(one_level_case_dirs / "*" / "K_ff.mtx"),
        reader=archive_source.readers().generic,
        archive_source=archive_source,
    )

    assert stream.sample_ids == (0, 1, 2)
    sample = stream.load_sample(1, matrix_format=MatrixFormat.CSR)
    np.testing.assert_array_equal(csr_array(sample).toarray(), _marker_matrix(1.0).toarray())


def test_recursive_wildcard_resolves_nested_directories(
    nested_case_dirs: Path, archive_source: InMemoryArchiveSource
) -> None:
    stream = GlobMatrixStream(
        str(nested_case_dirs / "**" / "K_ff.mtx"),
        reader=archive_source.readers().generic,
        archive_source=archive_source,
    )

    assert stream.sample_ids == (0, 1)


def test_one_level_wildcard_include_indices_keeps_directory_derived_ids(
    one_level_case_dirs: Path, archive_source: InMemoryArchiveSource
) -> None:
    stream = GlobMatrixStream(
        str(one_level_case_dirs / "*" / "K_ff.mtx"),
        include_indices=(0, 2),
        reader=archive_source.readers().generic,
        archive_source=archive_source,
    )

    assert stream.sample_ids == (0, 2)


def test_one_level_wildcard_custom_regex_on_directory_name(
    prefixed_case_dirs: Path, archive_source: InMemoryArchiveSource
) -> None:
    """A non-numeric-only directory name (``case_3``) still yields a usable id."""
    stream = GlobMatrixStream(
        str(prefixed_case_dirs / "*" / "K_ff.mtx"),
        sample_id_regex=r"case_(\d+)",
        reader=archive_source.readers().generic,
        archive_source=archive_source,
    )

    assert stream.sample_ids == (3, 7)


def test_flat_filename_glob_is_unaffected_by_directory_id_support(
    flat_filename_dir: Path, archive_source: InMemoryArchiveSource
) -> None:
    """Regression: a flat A_*.txt layout still derives ids from the filename, not a directory."""
    stream = GlobMatrixStream(
        str(flat_filename_dir / "A_*.txt"),
        reader=archive_source.readers().generic,
        archive_source=archive_source,
    )

    assert stream.sample_ids == (5, 9)
