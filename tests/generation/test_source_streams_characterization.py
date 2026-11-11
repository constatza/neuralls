"""Characterization tests pinning current source-stream behavior.

These lock in the observable behavior of the matrix/vector stream classes —
which files a glob matches, the sample ids derived from filenames, iteration
order, per-source shape validation, and the divergences between the glob and
single-file variants — so structural refactors of ``source_streams`` cannot
silently change any of it.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.source_streams import (
    EnumerateBy,
    GlobMatrixStream,
    GlobVectorStream,
    NpyMatrixStream,
    NpyVectorStream,
    TxtMatrixStream,
    TxtVectorStream,
    open_matrix_stream,
    open_vector_stream,
)
from neuralls.platform.storage.matrix_readers import read_matrix
from tests.fakes.archive_source import InMemoryArchiveSource

ARCHIVE_DIR = Path("/archive")


def _marker_matrix(value: float) -> np.ndarray:
    """Return a 2x2 matrix whose contents identify one source file."""
    return np.array([[value, 0.0], [0.0, value + 1.0]], dtype=np.float64)


def _marker_vector(value: float) -> np.ndarray:
    """Return a length-3 vector whose contents identify one source file."""
    return np.array([value, value + 1.0, value + 2.0], dtype=np.float64)


@pytest.fixture
def archive() -> InMemoryArchiveSource:
    """An empty in-memory archive; each test registers the files it reads."""
    return InMemoryArchiveSource()


@pytest.fixture
def readers(archive: InMemoryArchiveSource) -> MatrixReaders:
    """Readers over the arrays registered in ``archive``."""
    return archive.readers()


@pytest.fixture
def digit_named_matrix_dir(archive: InMemoryArchiveSource) -> Path:
    """Matrix .txt files whose stems end in non-contiguous, unsorted integer ids."""
    mat_dir = ARCHIVE_DIR / "digit_matrices"
    for stem, value in (("A_2", 2.0), ("A_10", 10.0), ("A_007", 7.0)):
        archive.add(mat_dir / f"{stem}.txt", _marker_matrix(value))
    return mat_dir


@pytest.fixture
def digit_named_vector_dir(archive: InMemoryArchiveSource) -> Path:
    """Vector .txt files whose stems end in non-contiguous, unsorted integer ids."""
    vec_dir = ARCHIVE_DIR / "digit_vectors"
    for stem, value in (("b_2", 2.0), ("b_10", 10.0), ("b_007", 7.0)):
        archive.add(vec_dir / f"{stem}.txt", _marker_vector(value))
    return vec_dir


@pytest.fixture
def mixed_extension_matrix_dir(archive: InMemoryArchiveSource) -> Path:
    """A single glob directory holding both .npy and .txt matrix files."""
    mat_dir = ARCHIVE_DIR / "mixed_matrices"
    archive.add(mat_dir / "A_1.txt", _marker_matrix(1.0))
    archive.add(mat_dir / "A_2.npy", _marker_matrix(2.0))
    return mat_dir


@pytest.fixture
def mixed_extension_vector_dir(archive: InMemoryArchiveSource) -> Path:
    """A single glob directory holding both .npy and .txt vector files."""
    vec_dir = ARCHIVE_DIR / "mixed_vectors"
    archive.add(vec_dir / "b_1.txt", _marker_vector(1.0))
    archive.add(vec_dir / "b_2.npy", _marker_vector(2.0))
    return vec_dir


@pytest.fixture
def prefixed_matrix_dir(archive: InMemoryArchiveSource) -> Path:
    """Matrix files whose ids are the leading digits of the stem."""
    mat_dir = ARCHIVE_DIR / "prefixed"
    archive.add(mat_dir / "12_case_3.txt", _marker_matrix(12.0))
    archive.add(mat_dir / "40_case_3.txt", _marker_matrix(40.0))
    return mat_dir


@pytest.fixture
def prefixed_vector_dir(archive: InMemoryArchiveSource) -> Path:
    """Vector files whose ids are the leading digits of the stem."""
    vec_dir = ARCHIVE_DIR / "prefixed_vectors"
    archive.add(vec_dir / "12_case_3.txt", _marker_vector(12.0))
    archive.add(vec_dir / "40_case_3.txt", _marker_vector(40.0))
    return vec_dir


@pytest.fixture
def colliding_matrix_dir(archive: InMemoryArchiveSource) -> Path:
    """Two matrix files whose stems end in the same id."""
    mat_dir = ARCHIVE_DIR / "dupes"
    archive.add(mat_dir / "A_1.txt", _marker_matrix(1.0))
    archive.add(mat_dir / "B_1.txt", _marker_matrix(2.0))
    return mat_dir


@pytest.fixture
def colliding_vector_dir(archive: InMemoryArchiveSource) -> Path:
    """Two vector files whose stems end in the same id."""
    vec_dir = ARCHIVE_DIR / "vector_dupes"
    archive.add(vec_dir / "a_1.txt", _marker_vector(1.0))
    archive.add(vec_dir / "b_1.txt", _marker_vector(2.0))
    return vec_dir


@pytest.fixture
def digitless_matrix_dir(archive: InMemoryArchiveSource) -> Path:
    """A matrix file whose stem holds no digits."""
    mat_dir = ARCHIVE_DIR / "no_digits"
    archive.add(mat_dir / "alpha.txt", _marker_matrix(1.0))
    return mat_dir


@pytest.fixture
def digitless_vector_dir(archive: InMemoryArchiveSource) -> Path:
    """A vector file whose stem holds no digits."""
    vec_dir = ARCHIVE_DIR / "no_digit_vectors"
    archive.add(vec_dir / "alpha.txt", _marker_vector(1.0))
    return vec_dir


# ---------------------------------------------------------------------------
# Glob discovery: sample-id derivation and ordering
# ---------------------------------------------------------------------------


def test_glob_matrix_stream_derives_ids_from_trailing_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    """The default regex takes the last digit run of the stem, ids stay unsorted-file-order free."""
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"), reader=readers.generic, archive_source=archive
    )

    assert stream.sample_ids == (2, 7, 10)
    np.testing.assert_array_equal(stream.load_dense_sample(7).matrix, _marker_matrix(7.0))
    np.testing.assert_array_equal(stream.load_dense_sample(10).matrix, _marker_matrix(10.0))


def test_glob_vector_stream_derives_ids_from_trailing_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"), reader=readers.generic, archive_source=archive
    )

    assert stream.sample_ids == (2, 7, 10)
    np.testing.assert_array_equal(stream.load_sample(7).vector, _marker_vector(7.0))
    np.testing.assert_array_equal(stream.load_sample(10).vector, _marker_vector(10.0))


def test_glob_matrix_stream_custom_regex_selects_leading_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, prefixed_matrix_dir: Path
) -> None:
    stream = GlobMatrixStream(
        str(prefixed_matrix_dir / "*_case_3.txt"),
        sample_id_regex=r"(\d+)",
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (12, 40)
    np.testing.assert_array_equal(stream.load_dense_sample(12).matrix, _marker_matrix(12.0))


def test_glob_vector_stream_custom_regex_selects_leading_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, prefixed_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(prefixed_vector_dir / "*_case_3.txt"),
        sample_id_regex=r"(\d+)",
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (12, 40)
    np.testing.assert_array_equal(stream.load_sample(12).vector, _marker_vector(12.0))


def test_glob_matrix_stream_rejects_filename_without_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digitless_matrix_dir: Path
) -> None:
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(digitless_matrix_dir / "*.txt"), reader=readers.generic, archive_source=archive
        )


def test_glob_vector_stream_rejects_filename_without_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digitless_vector_dir: Path
) -> None:
    with pytest.raises(ValueError):
        GlobVectorStream(
            str(digitless_vector_dir / "*.txt"), reader=readers.generic, archive_source=archive
        )


def test_glob_matrix_stream_rejects_duplicate_sample_ids(
    archive: InMemoryArchiveSource, readers: MatrixReaders, colliding_matrix_dir: Path
) -> None:
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(colliding_matrix_dir / "*_1.txt"), reader=readers.generic, archive_source=archive
        )


def test_glob_vector_stream_rejects_duplicate_sample_ids(
    archive: InMemoryArchiveSource, readers: MatrixReaders, colliding_vector_dir: Path
) -> None:
    with pytest.raises(ValueError):
        GlobVectorStream(
            str(colliding_vector_dir / "*_1.txt"), reader=readers.generic, archive_source=archive
        )


def test_glob_matrix_stream_missing_parent_directory_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    with pytest.raises(FileNotFoundError):
        GlobMatrixStream(
            str(ARCHIVE_DIR / "absent" / "A_*.txt"),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_vector_stream_missing_parent_directory_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    with pytest.raises(FileNotFoundError):
        GlobVectorStream(
            str(ARCHIVE_DIR / "absent" / "b_*.txt"),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_matrix_stream_without_matches_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        GlobMatrixStream(
            str(digit_named_matrix_dir / "Z_*.txt"),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_vector_stream_without_matches_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        GlobVectorStream(
            str(digit_named_vector_dir / "z_*.txt"),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_matrix_stream_enumerate_by_overrides_filename_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    """enumerate_by renumbers by sorted name, ignoring the ids the filenames encode."""
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"),
        enumerate_by=EnumerateBy.NAME,
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)
    # Sorted names are A_007.txt, A_10.txt, A_2.txt.
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(7.0))
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(10.0))
    np.testing.assert_array_equal(stream.load_dense_sample(2).matrix, _marker_matrix(2.0))


def test_glob_vector_stream_enumerate_by_overrides_filename_digits(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"),
        enumerate_by=EnumerateBy.NAME,
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(7.0))
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(10.0))
    np.testing.assert_array_equal(stream.load_sample(2).vector, _marker_vector(2.0))


def test_glob_matrix_stream_include_indices_keeps_regex_derived_ids(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    """Filtering never renumbers: filename-derived ids survive include_indices."""
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"),
        include_indices=(10, 2),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (2, 10)
    np.testing.assert_array_equal(stream.load_dense_sample(10).matrix, _marker_matrix(10.0))


def test_glob_vector_stream_exclude_indices_keeps_regex_derived_ids(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"),
        exclude_indices=(7,),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (2, 10)


def test_glob_vector_stream_rejects_emptying_all_samples(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    with pytest.raises(ValueError):
        GlobVectorStream(
            str(digit_named_vector_dir / "b_*.txt"),
            exclude_indices=(2, 7, 10),
            reader=readers.generic,
            archive_source=archive,
        )


# ---------------------------------------------------------------------------
# Glob loading: per-extension handling and iteration order
# ---------------------------------------------------------------------------


def test_glob_matrix_stream_loads_npy_and_txt_from_one_glob(
    archive: InMemoryArchiveSource, readers: MatrixReaders, mixed_extension_matrix_dir: Path
) -> None:
    stream = GlobMatrixStream(
        str(mixed_extension_matrix_dir / "A_*"), reader=readers.generic, archive_source=archive
    )

    assert stream.sample_ids == (1, 2)
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(1.0))
    np.testing.assert_array_equal(stream.load_dense_sample(2).matrix, _marker_matrix(2.0))


def test_glob_vector_stream_loads_npy_and_txt_from_one_glob(
    archive: InMemoryArchiveSource, readers: MatrixReaders, mixed_extension_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(mixed_extension_vector_dir / "b_*"), reader=readers.generic, archive_source=archive
    )

    assert stream.sample_ids == (1, 2)
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(1.0))
    np.testing.assert_array_equal(stream.load_sample(2).vector, _marker_vector(2.0))


def test_glob_matrix_stream_rejects_unsupported_extension(
    archive: InMemoryArchiveSource,
) -> None:
    """The suffix check is the real reader's; it fails before reading any bytes."""
    archive.add(ARCHIVE_DIR / "csv_matrices" / "A_1.csv", _marker_matrix(1.0))
    stream = GlobMatrixStream(
        str(ARCHIVE_DIR / "csv_matrices" / "A_*"),
        reader=partial(read_matrix, lazy=True),
        archive_source=archive,
    )

    with pytest.raises(ValueError):
        stream.load_dense_sample(1)


def test_glob_vector_stream_rejects_unsupported_extension(
    archive: InMemoryArchiveSource,
) -> None:
    """The suffix check is the real reader's; it fails before reading any bytes."""
    archive.add(ARCHIVE_DIR / "csv_vectors" / "b_1.csv", _marker_vector(1.0))
    stream = GlobVectorStream(
        str(ARCHIVE_DIR / "csv_vectors" / "b_*"),
        reader=partial(read_matrix, lazy=True),
        archive_source=archive,
    )

    with pytest.raises(ValueError):
        stream.load_sample(1)


def test_glob_matrix_stream_rejects_non_2d_npy_file(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    archive.add(
        ARCHIVE_DIR / "stacked" / "A_1.npy",
        np.stack([_marker_matrix(1.0), _marker_matrix(2.0)]),
    )
    stream = GlobMatrixStream(
        str(ARCHIVE_DIR / "stacked" / "A_*.npy"), reader=readers.generic, archive_source=archive
    )

    with pytest.raises(ValueError):
        stream.load_dense_sample(1)


def test_glob_matrix_stream_rejects_non_2d_txt_file(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    archive.add(ARCHIVE_DIR / "flat" / "A_1.txt", np.array([1.0, 2.0, 3.0]))
    stream = GlobMatrixStream(
        str(ARCHIVE_DIR / "flat" / "A_*.txt"), reader=readers.generic, archive_source=archive
    )

    with pytest.raises(ValueError):
        stream.load_dense_sample(1)


def test_glob_vector_stream_reshapes_2d_column_npy_to_1d(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    """A glob-matched (n,1) .npy is one column vector, not a stack of n scalars."""
    archive.add(ARCHIVE_DIR / "column_vectors" / "b_1.npy", _marker_vector(1.0).reshape(3, 1))
    stream = GlobVectorStream(
        str(ARCHIVE_DIR / "column_vectors" / "b_*.npy"),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (1,)
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(1.0))


def test_glob_vector_stream_rejects_genuinely_2d_npy(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    archive.add(
        ARCHIVE_DIR / "matrix_shaped_vectors" / "b_1.npy", np.ones((3, 2), dtype=np.float64)
    )
    stream = GlobVectorStream(
        str(ARCHIVE_DIR / "matrix_shaped_vectors" / "b_*.npy"),
        reader=readers.generic,
        archive_source=archive,
    )

    with pytest.raises(ValueError):
        stream.load_sample(1)


def test_glob_matrix_stream_unknown_sample_id_raises_key_error(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"), reader=readers.generic, archive_source=archive
    )

    with pytest.raises(KeyError):
        stream.load_dense_sample(99)


def test_glob_vector_stream_unknown_sample_id_raises_key_error(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"), reader=readers.generic, archive_source=archive
    )

    with pytest.raises(KeyError):
        stream.load_sample(99)


def test_glob_matrix_stream_iterates_in_sorted_sample_id_order(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_matrix_dir: Path
) -> None:
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"), reader=readers.generic, archive_source=archive
    )

    dense = list(stream.iter_dense_samples())
    sparse = list(stream.iter_sparse_samples())

    assert [sample.sample_id for sample in dense] == [2, 7, 10]
    assert [sample.sample_id for sample in sparse] == [2, 7, 10]
    np.testing.assert_array_equal(dense[1].matrix, _marker_matrix(7.0))


def test_glob_vector_stream_iterates_in_sorted_sample_id_order(
    archive: InMemoryArchiveSource, readers: MatrixReaders, digit_named_vector_dir: Path
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"), reader=readers.generic, archive_source=archive
    )

    samples = list(stream.iter_samples())

    assert [sample.sample_id for sample in samples] == [2, 7, 10]
    np.testing.assert_array_equal(samples[1].vector, _marker_vector(7.0))


def test_glob_matrix_stream_sparse_components_reconstruct_dense(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    dense = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    archive.add(ARCHIVE_DIR / "sparse_source" / "A_1.txt", dense)

    sample = GlobMatrixStream(
        str(ARCHIVE_DIR / "sparse_source" / "A_*.txt"),
        reader=readers.generic,
        archive_source=archive,
    ).load_sparse_sample(1)

    assert sample.size == (2, 2)
    reconstructed = np.zeros(sample.size, dtype=np.float64)
    reconstructed[sample.indices[0], sample.indices[1]] = sample.values
    np.testing.assert_array_equal(reconstructed, dense)


# ---------------------------------------------------------------------------
# Single-file streams
# ---------------------------------------------------------------------------


def test_npy_matrix_stream_single_matrix_exposes_one_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "matrix.npy", _marker_matrix(1.0))

    stream = NpyMatrixStream(path, reader=readers.dense)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(1.0))
    with pytest.raises(KeyError):
        stream.load_dense_sample(1)


def test_npy_matrix_stream_stack_iterates_every_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(
        ARCHIVE_DIR / "stack.npy", np.stack([_marker_matrix(1.0), _marker_matrix(5.0)])
    )

    stream = NpyMatrixStream(path, reader=readers.dense)

    assert stream.sample_ids == (0, 1)
    assert [sample.sample_id for sample in stream.iter_dense_samples()] == [0, 1]
    assert [sample.sample_id for sample in stream.iter_sparse_samples()] == [0, 1]
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(5.0))


def test_npy_matrix_stream_rejects_4d_array(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "hyper.npy", np.ones((2, 2, 2, 2), dtype=np.float64))

    with pytest.raises(ValueError):
        NpyMatrixStream(path, reader=readers.dense)


def test_txt_matrix_stream_loads_the_only_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "matrix.txt", _marker_matrix(1.0))

    stream = TxtMatrixStream(path, reader=readers.generic)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(1.0))
    assert [sample.sample_id for sample in stream.iter_dense_samples()] == [0]
    assert [sample.sample_id for sample in stream.iter_sparse_samples()] == [0]
    with pytest.raises(KeyError):
        stream.load_dense_sample(3)


def test_txt_matrix_stream_rejects_1d_file(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "flat.txt", np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        TxtMatrixStream(path, reader=readers.generic).load_dense_sample(0)


def test_npy_vector_stream_treats_2d_array_as_a_stack(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    """A single-file (n,1) .npy is a stack of n length-1 vectors — unlike the glob variant."""
    path = archive.add(ARCHIVE_DIR / "column.npy", _marker_vector(1.0).reshape(3, 1))

    stream = NpyVectorStream(path, reader=readers.dense)

    assert stream.sample_ids == (0, 1, 2)
    np.testing.assert_array_equal(stream.load_sample(0).vector, np.array([1.0]))


def test_npy_vector_stream_single_vector_exposes_one_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "vector.npy", _marker_vector(1.0))

    stream = NpyVectorStream(path, reader=readers.dense)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(1.0))
    assert [sample.sample_id for sample in stream.iter_samples()] == [0]
    with pytest.raises(KeyError):
        stream.load_sample(1)


def test_npy_vector_stream_rejects_3d_array(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "cube.npy", np.ones((2, 2, 2), dtype=np.float64))

    with pytest.raises(ValueError):
        NpyVectorStream(path, reader=readers.dense)


def test_txt_vector_stream_loads_the_only_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "vector.txt", _marker_vector(1.0))

    stream = TxtVectorStream(path, reader=readers.generic)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(1.0))
    assert [sample.sample_id for sample in stream.iter_samples()] == [0]
    with pytest.raises(KeyError):
        stream.load_sample(3)


def test_txt_vector_stream_rejects_2d_file(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "matrix.txt", np.ones((3, 2), dtype=np.float64))

    with pytest.raises(ValueError):
        TxtVectorStream(path, reader=readers.generic).load_sample(0)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def matrix_expressions(archive: InMemoryArchiveSource) -> tuple[str, str, str]:
    """Path expressions for a single .npy, a single .txt and a one-file glob of matrices."""
    npy_path = archive.add(ARCHIVE_DIR / "matrix.npy", _marker_matrix(1.0))
    txt_path = archive.add(ARCHIVE_DIR / "matrix.txt", _marker_matrix(1.0))
    archive.add(ARCHIVE_DIR / "globbed" / "A_1.txt", _marker_matrix(1.0))
    return str(npy_path), str(txt_path), str(ARCHIVE_DIR / "globbed" / "A_*.txt")


@pytest.fixture
def vector_expressions(archive: InMemoryArchiveSource) -> tuple[str, str, str]:
    """Path expressions for a single .npy, a single .txt and a one-file glob of vectors."""
    npy_path = archive.add(ARCHIVE_DIR / "vector.npy", _marker_vector(1.0))
    txt_path = archive.add(ARCHIVE_DIR / "vector.txt", _marker_vector(1.0))
    archive.add(ARCHIVE_DIR / "globbed_vectors" / "b_1.txt", _marker_vector(1.0))
    return str(npy_path), str(txt_path), str(ARCHIVE_DIR / "globbed_vectors" / "b_*.txt")


def test_open_matrix_stream_selects_stream_type_by_expression(
    archive: InMemoryArchiveSource, readers: MatrixReaders, matrix_expressions: tuple[str, str, str]
) -> None:
    npy_expr, txt_expr, glob_expr = matrix_expressions

    assert isinstance(
        open_matrix_stream(npy_expr, readers=readers, archive_source=archive), NpyMatrixStream
    )
    assert isinstance(
        open_matrix_stream(txt_expr, readers=readers, archive_source=archive), TxtMatrixStream
    )
    assert isinstance(
        open_matrix_stream(glob_expr, readers=readers, archive_source=archive), GlobMatrixStream
    )


def test_open_vector_stream_selects_stream_type_by_expression(
    archive: InMemoryArchiveSource, readers: MatrixReaders, vector_expressions: tuple[str, str, str]
) -> None:
    npy_expr, txt_expr, glob_expr = vector_expressions

    assert isinstance(
        open_vector_stream(npy_expr, readers=readers, archive_source=archive), NpyVectorStream
    )
    assert isinstance(
        open_vector_stream(txt_expr, readers=readers, archive_source=archive), TxtVectorStream
    )
    assert isinstance(
        open_vector_stream(glob_expr, readers=readers, archive_source=archive), GlobVectorStream
    )


def test_open_matrix_stream_rejects_unknown_suffix(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "matrix.csv", _marker_matrix(1.0))

    with pytest.raises(ValueError):
        open_matrix_stream(str(path), readers=readers, archive_source=archive)


def test_open_vector_stream_rejects_unknown_suffix(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "vector.csv", _marker_vector(1.0))

    with pytest.raises(ValueError):
        open_vector_stream(str(path), readers=readers, archive_source=archive)


def test_open_matrix_stream_missing_file_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    with pytest.raises(FileNotFoundError):
        open_matrix_stream(str(ARCHIVE_DIR / "absent.npy"), readers=readers, archive_source=archive)


def test_open_vector_stream_missing_file_raises(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    with pytest.raises(FileNotFoundError):
        open_vector_stream(str(ARCHIVE_DIR / "absent.npy"), readers=readers, archive_source=archive)


def test_open_vector_stream_include_indices_requires_glob_source(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "vector.npy", _marker_vector(1.0))

    with pytest.raises(ValueError):
        open_vector_stream(str(path), include_indices=(0,), readers=readers, archive_source=archive)
