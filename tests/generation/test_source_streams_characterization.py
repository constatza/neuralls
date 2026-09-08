"""Characterization tests pinning current source-stream behavior.

These lock in the observable behavior of the matrix/vector stream classes —
which files a glob matches, the sample ids derived from filenames, iteration
order, per-source shape validation, and the divergences between the glob and
single-file variants — so structural refactors of ``source_streams`` cannot
silently change any of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def _marker_matrix(value: float) -> np.ndarray:
    """Return a 2x2 matrix whose contents identify one source file."""
    return np.array([[value, 0.0], [0.0, value + 1.0]], dtype=np.float64)


def _marker_vector(value: float) -> np.ndarray:
    """Return a length-3 vector whose contents identify one source file."""
    return np.array([value, value + 1.0, value + 2.0], dtype=np.float64)


@pytest.fixture
def digit_named_matrix_dir(tmp_path: Path) -> Path:
    """Matrix .txt files whose stems end in non-contiguous, unsorted integer ids."""
    mat_dir = tmp_path / "digit_matrices"
    mat_dir.mkdir()
    for stem, value in (("A_2", 2.0), ("A_10", 10.0), ("A_007", 7.0)):
        np.savetxt(mat_dir / f"{stem}.txt", _marker_matrix(value))
    return mat_dir


@pytest.fixture
def digit_named_vector_dir(tmp_path: Path) -> Path:
    """Vector .txt files whose stems end in non-contiguous, unsorted integer ids."""
    vec_dir = tmp_path / "digit_vectors"
    vec_dir.mkdir()
    for stem, value in (("b_2", 2.0), ("b_10", 10.0), ("b_007", 7.0)):
        np.savetxt(vec_dir / f"{stem}.txt", _marker_vector(value))
    return vec_dir


@pytest.fixture
def mixed_extension_matrix_dir(tmp_path: Path) -> Path:
    """A single glob directory holding both .npy and .txt matrix files."""
    mat_dir = tmp_path / "mixed_matrices"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "A_1.txt", _marker_matrix(1.0))
    np.save(mat_dir / "A_2.npy", _marker_matrix(2.0))
    return mat_dir


@pytest.fixture
def mixed_extension_vector_dir(tmp_path: Path) -> Path:
    """A single glob directory holding both .npy and .txt vector files."""
    vec_dir = tmp_path / "mixed_vectors"
    vec_dir.mkdir()
    np.savetxt(vec_dir / "b_1.txt", _marker_vector(1.0))
    np.save(vec_dir / "b_2.npy", _marker_vector(2.0))
    return vec_dir


# ---------------------------------------------------------------------------
# Glob discovery: sample-id derivation and ordering
# ---------------------------------------------------------------------------


def test_glob_matrix_stream_derives_ids_from_trailing_digits(
    digit_named_matrix_dir: Path,
) -> None:
    """The default regex takes the last digit run of the stem, ids stay unsorted-file-order free."""
    stream = GlobMatrixStream(str(digit_named_matrix_dir / "A_*.txt"))

    assert stream.sample_ids == (2, 7, 10)
    np.testing.assert_array_equal(stream.load_dense_sample(7).matrix, _marker_matrix(7.0))
    np.testing.assert_array_equal(stream.load_dense_sample(10).matrix, _marker_matrix(10.0))


def test_glob_vector_stream_derives_ids_from_trailing_digits(
    digit_named_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(str(digit_named_vector_dir / "b_*.txt"))

    assert stream.sample_ids == (2, 7, 10)
    np.testing.assert_array_equal(stream.load_sample(7).vector, _marker_vector(7.0))
    np.testing.assert_array_equal(stream.load_sample(10).vector, _marker_vector(10.0))


def test_glob_matrix_stream_custom_regex_selects_leading_digits(tmp_path: Path) -> None:
    mat_dir = tmp_path / "prefixed"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "12_case_3.txt", _marker_matrix(12.0))
    np.savetxt(mat_dir / "40_case_3.txt", _marker_matrix(40.0))

    stream = GlobMatrixStream(str(mat_dir / "*_case_3.txt"), sample_id_regex=r"(\d+)")

    assert stream.sample_ids == (12, 40)
    np.testing.assert_array_equal(stream.load_dense_sample(12).matrix, _marker_matrix(12.0))


def test_glob_vector_stream_custom_regex_selects_leading_digits(tmp_path: Path) -> None:
    vec_dir = tmp_path / "prefixed_vectors"
    vec_dir.mkdir()
    np.savetxt(vec_dir / "12_case_3.txt", _marker_vector(12.0))
    np.savetxt(vec_dir / "40_case_3.txt", _marker_vector(40.0))

    stream = GlobVectorStream(str(vec_dir / "*_case_3.txt"), sample_id_regex=r"(\d+)")

    assert stream.sample_ids == (12, 40)
    np.testing.assert_array_equal(stream.load_sample(12).vector, _marker_vector(12.0))


def test_glob_matrix_stream_rejects_filename_without_digits(tmp_path: Path) -> None:
    mat_dir = tmp_path / "no_digits"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "alpha.txt", _marker_matrix(1.0))

    with pytest.raises(ValueError, match="Could not extract sample id"):
        GlobMatrixStream(str(mat_dir / "*.txt"))


def test_glob_vector_stream_rejects_filename_without_digits(tmp_path: Path) -> None:
    vec_dir = tmp_path / "no_digit_vectors"
    vec_dir.mkdir()
    np.savetxt(vec_dir / "alpha.txt", _marker_vector(1.0))

    with pytest.raises(ValueError, match="Could not extract sample id"):
        GlobVectorStream(str(vec_dir / "*.txt"))


def test_glob_matrix_stream_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    mat_dir = tmp_path / "dupes"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "A_1.txt", _marker_matrix(1.0))
    np.savetxt(mat_dir / "B_1.txt", _marker_matrix(2.0))

    with pytest.raises(ValueError, match="Duplicate matrix sample id 1"):
        GlobMatrixStream(str(mat_dir / "*_1.txt"))


def test_glob_vector_stream_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    vec_dir = tmp_path / "vector_dupes"
    vec_dir.mkdir()
    np.savetxt(vec_dir / "a_1.txt", _marker_vector(1.0))
    np.savetxt(vec_dir / "b_1.txt", _marker_vector(2.0))

    with pytest.raises(ValueError, match="Duplicate vector sample id 1"):
        GlobVectorStream(str(vec_dir / "*_1.txt"))


def test_glob_matrix_stream_missing_parent_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Matrix glob parent directory not found"):
        GlobMatrixStream(str(tmp_path / "absent" / "A_*.txt"))


def test_glob_vector_stream_missing_parent_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Vector glob parent directory not found"):
        GlobVectorStream(str(tmp_path / "absent" / "b_*.txt"))


def test_glob_matrix_stream_without_matches_raises(digit_named_matrix_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No matrix files match glob"):
        GlobMatrixStream(str(digit_named_matrix_dir / "Z_*.txt"))


def test_glob_vector_stream_without_matches_raises(digit_named_vector_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No vector files match glob"):
        GlobVectorStream(str(digit_named_vector_dir / "z_*.txt"))


def test_glob_matrix_stream_enumerate_by_overrides_filename_digits(
    digit_named_matrix_dir: Path,
) -> None:
    """enumerate_by renumbers by sorted name, ignoring the ids the filenames encode."""
    stream = GlobMatrixStream(
        str(digit_named_matrix_dir / "A_*.txt"), enumerate_by=EnumerateBy.NAME
    )

    assert stream.sample_ids == (0, 1, 2)
    # Sorted names are A_007.txt, A_10.txt, A_2.txt.
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(7.0))
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(10.0))
    np.testing.assert_array_equal(stream.load_dense_sample(2).matrix, _marker_matrix(2.0))


def test_glob_vector_stream_enumerate_by_overrides_filename_digits(
    digit_named_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(
        str(digit_named_vector_dir / "b_*.txt"), enumerate_by=EnumerateBy.NAME
    )

    assert stream.sample_ids == (0, 1, 2)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(7.0))
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(10.0))
    np.testing.assert_array_equal(stream.load_sample(2).vector, _marker_vector(2.0))


def test_glob_matrix_stream_include_indices_keeps_regex_derived_ids(
    digit_named_matrix_dir: Path,
) -> None:
    """Filtering never renumbers: filename-derived ids survive include_indices."""
    stream = GlobMatrixStream(str(digit_named_matrix_dir / "A_*.txt"), include_indices=(10, 2))

    assert stream.sample_ids == (2, 10)
    np.testing.assert_array_equal(stream.load_dense_sample(10).matrix, _marker_matrix(10.0))


def test_glob_vector_stream_exclude_indices_keeps_regex_derived_ids(
    digit_named_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(str(digit_named_vector_dir / "b_*.txt"), exclude_indices=(7,))

    assert stream.sample_ids == (2, 10)


def test_glob_vector_stream_rejects_emptying_all_samples(digit_named_vector_dir: Path) -> None:
    with pytest.raises(ValueError, match="No vector samples remain after filtering glob"):
        GlobVectorStream(str(digit_named_vector_dir / "b_*.txt"), exclude_indices=(2, 7, 10))


# ---------------------------------------------------------------------------
# Glob loading: per-extension handling and iteration order
# ---------------------------------------------------------------------------


def test_glob_matrix_stream_loads_npy_and_txt_from_one_glob(
    mixed_extension_matrix_dir: Path,
) -> None:
    stream = GlobMatrixStream(str(mixed_extension_matrix_dir / "A_*"))

    assert stream.sample_ids == (1, 2)
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(1.0))
    np.testing.assert_array_equal(stream.load_dense_sample(2).matrix, _marker_matrix(2.0))


def test_glob_vector_stream_loads_npy_and_txt_from_one_glob(
    mixed_extension_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(str(mixed_extension_vector_dir / "b_*"))

    assert stream.sample_ids == (1, 2)
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(1.0))
    np.testing.assert_array_equal(stream.load_sample(2).vector, _marker_vector(2.0))


def test_glob_matrix_stream_rejects_unsupported_extension(tmp_path: Path) -> None:
    mat_dir = tmp_path / "csv_matrices"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "A_1.csv", _marker_matrix(1.0), delimiter=",")

    stream = GlobMatrixStream(str(mat_dir / "A_*"))

    with pytest.raises(ValueError, match="Unsupported matrix file extension in glob"):
        stream.load_dense_sample(1)


def test_glob_vector_stream_rejects_unsupported_extension(tmp_path: Path) -> None:
    vec_dir = tmp_path / "csv_vectors"
    vec_dir.mkdir()
    np.savetxt(vec_dir / "b_1.csv", _marker_vector(1.0), delimiter=",")

    stream = GlobVectorStream(str(vec_dir / "b_*"))

    with pytest.raises(ValueError, match="Unsupported vector file extension in glob"):
        stream.load_sample(1)


def test_glob_matrix_stream_rejects_non_2d_npy_file(tmp_path: Path) -> None:
    mat_dir = tmp_path / "stacked"
    mat_dir.mkdir()
    np.save(mat_dir / "A_1.npy", np.stack([_marker_matrix(1.0), _marker_matrix(2.0)]))

    stream = GlobMatrixStream(str(mat_dir / "A_*.npy"))

    with pytest.raises(ValueError, match="must be a single 2D matrix"):
        stream.load_dense_sample(1)


def test_glob_matrix_stream_rejects_non_2d_txt_file(tmp_path: Path) -> None:
    mat_dir = tmp_path / "flat"
    mat_dir.mkdir()
    np.savetxt(mat_dir / "A_1.txt", np.array([1.0, 2.0, 3.0]))

    stream = GlobMatrixStream(str(mat_dir / "A_*.txt"))

    with pytest.raises(ValueError, match="must be a single 2D matrix"):
        stream.load_dense_sample(1)


def test_glob_vector_stream_reshapes_2d_column_npy_to_1d(tmp_path: Path) -> None:
    """A glob-matched (n,1) .npy is one column vector, not a stack of n scalars."""
    vec_dir = tmp_path / "column_vectors"
    vec_dir.mkdir()
    np.save(vec_dir / "b_1.npy", _marker_vector(1.0).reshape(3, 1))

    stream = GlobVectorStream(str(vec_dir / "b_*.npy"))

    assert stream.sample_ids == (1,)
    np.testing.assert_array_equal(stream.load_sample(1).vector, _marker_vector(1.0))


def test_glob_vector_stream_rejects_genuinely_2d_npy(tmp_path: Path) -> None:
    vec_dir = tmp_path / "matrix_shaped_vectors"
    vec_dir.mkdir()
    np.save(vec_dir / "b_1.npy", np.ones((3, 2), dtype=np.float64))

    stream = GlobVectorStream(str(vec_dir / "b_*.npy"))

    with pytest.raises(ValueError, match="Expected vector from"):
        stream.load_sample(1)


def test_glob_matrix_stream_unknown_sample_id_raises_key_error(
    digit_named_matrix_dir: Path,
) -> None:
    stream = GlobMatrixStream(str(digit_named_matrix_dir / "A_*.txt"))

    with pytest.raises(KeyError, match="Unknown matrix sample id 99"):
        stream.load_dense_sample(99)


def test_glob_vector_stream_unknown_sample_id_raises_key_error(
    digit_named_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(str(digit_named_vector_dir / "b_*.txt"))

    with pytest.raises(KeyError, match="Unknown vector sample id 99"):
        stream.load_sample(99)


def test_glob_matrix_stream_iterates_in_sorted_sample_id_order(
    digit_named_matrix_dir: Path,
) -> None:
    stream = GlobMatrixStream(str(digit_named_matrix_dir / "A_*.txt"))

    dense = list(stream.iter_dense_samples())
    sparse = list(stream.iter_sparse_samples())

    assert [sample.sample_id for sample in dense] == [2, 7, 10]
    assert [sample.sample_id for sample in sparse] == [2, 7, 10]
    np.testing.assert_array_equal(dense[1].matrix, _marker_matrix(7.0))


def test_glob_vector_stream_iterates_in_sorted_sample_id_order(
    digit_named_vector_dir: Path,
) -> None:
    stream = GlobVectorStream(str(digit_named_vector_dir / "b_*.txt"))

    samples = list(stream.iter_samples())

    assert [sample.sample_id for sample in samples] == [2, 7, 10]
    np.testing.assert_array_equal(samples[1].vector, _marker_vector(7.0))


def test_glob_matrix_stream_sparse_components_reconstruct_dense(tmp_path: Path) -> None:
    mat_dir = tmp_path / "sparse_source"
    mat_dir.mkdir()
    dense = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    np.savetxt(mat_dir / "A_1.txt", dense)

    sample = GlobMatrixStream(str(mat_dir / "A_*.txt")).load_sparse_sample(1)

    assert sample.size == (2, 2)
    reconstructed = np.zeros(sample.size, dtype=np.float64)
    reconstructed[sample.indices[0], sample.indices[1]] = sample.values
    np.testing.assert_array_equal(reconstructed, dense)


# ---------------------------------------------------------------------------
# Single-file streams
# ---------------------------------------------------------------------------


def test_npy_matrix_stream_single_matrix_exposes_one_sample(tmp_path: Path) -> None:
    path = tmp_path / "matrix.npy"
    np.save(path, _marker_matrix(1.0))

    stream = NpyMatrixStream(path)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(1.0))
    with pytest.raises(KeyError, match="Unknown matrix sample id 1"):
        stream.load_dense_sample(1)


def test_npy_matrix_stream_stack_iterates_every_sample(tmp_path: Path) -> None:
    path = tmp_path / "stack.npy"
    np.save(path, np.stack([_marker_matrix(1.0), _marker_matrix(5.0)]))

    stream = NpyMatrixStream(path)

    assert stream.sample_ids == (0, 1)
    assert [sample.sample_id for sample in stream.iter_dense_samples()] == [0, 1]
    assert [sample.sample_id for sample in stream.iter_sparse_samples()] == [0, 1]
    np.testing.assert_array_equal(stream.load_dense_sample(1).matrix, _marker_matrix(5.0))


def test_npy_matrix_stream_rejects_4d_array(tmp_path: Path) -> None:
    path = tmp_path / "hyper.npy"
    np.save(path, np.ones((2, 2, 2, 2), dtype=np.float64))

    with pytest.raises(ValueError, match=r"must have shape \(n,n\) or \(N,n,n\)"):
        NpyMatrixStream(path)


def test_txt_matrix_stream_loads_the_only_sample(tmp_path: Path) -> None:
    path = tmp_path / "matrix.txt"
    np.savetxt(path, _marker_matrix(1.0))

    stream = TxtMatrixStream(path)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_dense_sample(0).matrix, _marker_matrix(1.0))
    assert [sample.sample_id for sample in stream.iter_dense_samples()] == [0]
    assert [sample.sample_id for sample in stream.iter_sparse_samples()] == [0]
    with pytest.raises(KeyError, match="Unknown matrix sample id 3"):
        stream.load_dense_sample(3)


def test_txt_matrix_stream_rejects_1d_file(tmp_path: Path) -> None:
    path = tmp_path / "flat.txt"
    np.savetxt(path, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="must be a single 2D matrix"):
        TxtMatrixStream(path).load_dense_sample(0)


def test_npy_vector_stream_treats_2d_array_as_a_stack(tmp_path: Path) -> None:
    """A single-file (n,1) .npy is a stack of n length-1 vectors — unlike the glob variant."""
    path = tmp_path / "column.npy"
    np.save(path, _marker_vector(1.0).reshape(3, 1))

    stream = NpyVectorStream(path)

    assert stream.sample_ids == (0, 1, 2)
    np.testing.assert_array_equal(stream.load_sample(0).vector, np.array([1.0]))


def test_npy_vector_stream_single_vector_exposes_one_sample(tmp_path: Path) -> None:
    path = tmp_path / "vector.npy"
    np.save(path, _marker_vector(1.0))

    stream = NpyVectorStream(path)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(1.0))
    assert [sample.sample_id for sample in stream.iter_samples()] == [0]
    with pytest.raises(KeyError, match="Unknown vector sample id 1"):
        stream.load_sample(1)


def test_npy_vector_stream_rejects_3d_array(tmp_path: Path) -> None:
    path = tmp_path / "cube.npy"
    np.save(path, np.ones((2, 2, 2), dtype=np.float64))

    with pytest.raises(ValueError, match=r"must have shape \(n,\) or \(N,n\)"):
        NpyVectorStream(path)


def test_txt_vector_stream_loads_the_only_sample(tmp_path: Path) -> None:
    path = tmp_path / "vector.txt"
    np.savetxt(path, _marker_vector(1.0))

    stream = TxtVectorStream(path)

    assert stream.sample_ids == (0,)
    np.testing.assert_array_equal(stream.load_sample(0).vector, _marker_vector(1.0))
    assert [sample.sample_id for sample in stream.iter_samples()] == [0]
    with pytest.raises(KeyError, match="Unknown vector sample id 3"):
        stream.load_sample(3)


def test_txt_vector_stream_rejects_2d_file(tmp_path: Path) -> None:
    path = tmp_path / "matrix.txt"
    np.savetxt(path, np.ones((3, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="Expected vector from"):
        TxtVectorStream(path).load_sample(0)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_open_matrix_stream_selects_stream_type_by_expression(tmp_path: Path) -> None:
    npy_path = tmp_path / "matrix.npy"
    np.save(npy_path, _marker_matrix(1.0))
    txt_path = tmp_path / "matrix.txt"
    np.savetxt(txt_path, _marker_matrix(1.0))
    glob_dir = tmp_path / "globbed"
    glob_dir.mkdir()
    np.savetxt(glob_dir / "A_1.txt", _marker_matrix(1.0))

    assert isinstance(open_matrix_stream(str(npy_path)), NpyMatrixStream)
    assert isinstance(open_matrix_stream(str(txt_path)), TxtMatrixStream)
    assert isinstance(open_matrix_stream(str(glob_dir / "A_*.txt")), GlobMatrixStream)


def test_open_vector_stream_selects_stream_type_by_expression(tmp_path: Path) -> None:
    npy_path = tmp_path / "vector.npy"
    np.save(npy_path, _marker_vector(1.0))
    txt_path = tmp_path / "vector.txt"
    np.savetxt(txt_path, _marker_vector(1.0))
    glob_dir = tmp_path / "globbed_vectors"
    glob_dir.mkdir()
    np.savetxt(glob_dir / "b_1.txt", _marker_vector(1.0))

    assert isinstance(open_vector_stream(str(npy_path)), NpyVectorStream)
    assert isinstance(open_vector_stream(str(txt_path)), TxtVectorStream)
    assert isinstance(open_vector_stream(str(glob_dir / "b_*.txt")), GlobVectorStream)


def test_open_matrix_stream_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    np.savetxt(path, _marker_matrix(1.0), delimiter=",")

    with pytest.raises(ValueError, match="Unsupported matrix source"):
        open_matrix_stream(str(path))


def test_open_vector_stream_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "vector.csv"
    np.savetxt(path, _marker_vector(1.0), delimiter=",")

    with pytest.raises(ValueError, match="Unsupported vector source"):
        open_vector_stream(str(path))


def test_open_matrix_stream_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Matrix source not found"):
        open_matrix_stream(str(tmp_path / "absent.npy"))


def test_open_vector_stream_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Vector source not found"):
        open_vector_stream(str(tmp_path / "absent.npy"))


def test_open_vector_stream_include_indices_requires_glob_source(tmp_path: Path) -> None:
    path = tmp_path / "vector.npy"
    np.save(path, _marker_vector(1.0))

    with pytest.raises(ValueError, match="require a glob vector source"):
        open_vector_stream(str(path), include_indices=(0,))
