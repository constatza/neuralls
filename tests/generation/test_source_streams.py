from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation.bindings import bind_sources
from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.sample_ids import _enumerate_files
from neuralls.domain.generation.source_streams import (
    EnumerateBy,
    GlobMatrixStream,
    GlobVectorStream,
    open_matrix_stream,
    open_vector_stream,
)
from neuralls.platform.storage.archive_source import FsArchiveSource
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import spd_matrix

ARCHIVE_DIR = Path("/archive")


_ARBITRARY_PARAMETER_SETS = [(3000, 78000), (5000, 100000), (1000, 50000)]


@pytest.fixture
def archive() -> InMemoryArchiveSource:
    """An empty in-memory archive; each test registers the files it reads."""
    return InMemoryArchiveSource()


@pytest.fixture
def readers(archive: InMemoryArchiveSource) -> MatrixReaders:
    """Readers over the arrays registered in ``archive``."""
    return archive.readers()


def test_open_matrix_stream_from_npy_stack(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    matrix_stack = np.array(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[4.0, 1.0], [1.0, 5.0]],
        ],
        dtype=np.float64,
    )
    matrix_path = archive.add(ARCHIVE_DIR / "matrices.npy", matrix_stack)

    stream = open_matrix_stream(str(matrix_path), readers=readers, archive_source=archive)
    assert stream.sample_ids == (0, 1)

    dense_one = stream.load_dense_sample(1)
    np.testing.assert_allclose(dense_one.matrix, matrix_stack[1])

    sparse_zero = stream.load_sparse_sample(0)
    reconstructed = np.zeros(sparse_zero.size, dtype=np.float64)
    reconstructed[sparse_zero.indices[0], sparse_zero.indices[1]] = sparse_zero.values
    np.testing.assert_allclose(reconstructed, matrix_stack[0])


def test_bind_sources_broadcasts_single_matrix_to_many_rhs() -> None:
    bindings = bind_sources(
        matrix_ids=(0,),
        rhs_ids=(0, 1, 2),
    )
    assert [item.matrix_sample_id for item in bindings] == [0, 0, 0]
    assert [item.rhs_sample_id for item in bindings] == [0, 1, 2]


def test_open_vector_stream_from_npy_stack(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs_path = archive.add(ARCHIVE_DIR / "rhs.npy", rhs)

    stream = open_vector_stream(str(rhs_path), readers=readers, archive_source=archive)
    assert stream.sample_ids == (0, 1)
    np.testing.assert_allclose(stream.load_sample(1).vector, rhs[1])


@pytest.fixture
def arbitrary_named_txt_matrices(archive: InMemoryArchiveSource) -> tuple[Path, int]:
    """Three 5×5 SPD matrices with parameter-encoded filenames (no sequential integer ID)."""
    rng = np.random.default_rng(99)
    mat_dir = ARCHIVE_DIR / "arbitrary_matrices"
    n = 5
    for e1, e2 in _ARBITRARY_PARAMETER_SETS:
        archive.add(mat_dir / f"E1_{e1}_E2_{e2}_matrix.txt", spd_matrix(rng, n))
    return mat_dir, n


@pytest.fixture
def arbitrary_named_txt_vectors(archive: InMemoryArchiveSource) -> Path:
    """Three length-5 vectors with parameter-encoded filenames (no sequential integer ID)."""
    rng = np.random.default_rng(42)
    vec_dir = ARCHIVE_DIR / "vecs"
    for e1, e2 in _ARBITRARY_PARAMETER_SETS:
        archive.add(vec_dir / f"E1_{e1}_E2_{e2}_rhs.txt", rng.standard_normal(5))
    return vec_dir


@pytest.fixture
def in_memory_subdomain_matrices(
    archive: InMemoryArchiveSource, subdomain_matrices: dict[str, np.ndarray]
) -> Path:
    """The subdomain matrices registered in the in-memory archive."""
    mat_dir = ARCHIVE_DIR / "subdomain_matrices"
    for name, matrix in subdomain_matrices.items():
        archive.add(mat_dir / name, matrix)
    return mat_dir


def test_enumerate_files_assigns_sequential_ids_in_name_order(
    archive: InMemoryArchiveSource,
) -> None:
    paths = [ARCHIVE_DIR / f"file_z_{suffix}.txt" for suffix in ["c", "a", "b"]]

    result = _enumerate_files(paths, EnumerateBy.NAME, archive)

    assert sorted(result.keys()) == [0, 1, 2]
    assert [result[i].name for i in range(3)] == [
        "file_z_a.txt",
        "file_z_b.txt",
        "file_z_c.txt",
    ]


@pytest.fixture
def staggered_files(archive: InMemoryArchiveSource) -> list[Path]:
    """Files c, a, b created in that order, each later than the last, in both clocks."""
    return [
        archive.add(ARCHIVE_DIR / f"file_{suffix}.txt", np.zeros(1), mtime=float(i), ctime=float(i))
        for i, suffix in enumerate(["c", "a", "b"])
    ]


def test_enumerate_files_assigns_sequential_ids_by_mtime(
    archive: InMemoryArchiveSource, staggered_files: list[Path]
) -> None:
    result = _enumerate_files(staggered_files, EnumerateBy.MTIME, archive)

    # Creation order was c, a, b — IDs should follow mtime order
    assert result[0].name == "file_c.txt"
    assert result[1].name == "file_a.txt"
    assert result[2].name == "file_b.txt"


def test_creation_time_rejects_stat_without_birthtime() -> None:
    """A stat result without st_birthtime (as on Linux) names the file and the alternatives."""
    stat_without_birthtime = os.stat_result((0o100644, 1, 1, 1, 0, 0, 0, 1.0, 2.0, 3.0))

    with (
        patch.object(Path, "stat", return_value=stat_without_birthtime),
        pytest.raises(ValueError) as excinfo,
    ):
        FsArchiveSource().creation_time(Path("a.txt"))

    assert str(excinfo.value) == (
        "enumerate_by='ctime' needs file creation times, which this platform does not "
        "record (a.txt). Use enumerate_by='mtime' or 'name', or sample_id_regex."
    )


def test_enumerate_files_assigns_sequential_ids_by_ctime(
    archive: InMemoryArchiveSource, staggered_files: list[Path]
) -> None:
    result = _enumerate_files(staggered_files, EnumerateBy.CTIME, archive)

    assert [result[i].name for i in range(3)] == ["file_c.txt", "file_a.txt", "file_b.txt"]


def test_enumerate_files_by_ctime_raises_without_creation_times(
    archive: InMemoryArchiveSource,
) -> None:
    paths = [archive.add(ARCHIVE_DIR / name, np.zeros(1), mtime=0.0) for name in ("a.txt", "b.txt")]

    with pytest.raises(ValueError, match="creation times"):
        _enumerate_files(paths, EnumerateBy.CTIME, archive)


def test_enumerate_files_breaks_timestamp_ties_by_name(archive: InMemoryArchiveSource) -> None:
    paths = [
        archive.add(ARCHIVE_DIR / f"file_{suffix}.txt", np.zeros(1), mtime=1_000_000.0)
        for suffix in ["c", "b", "a"]
    ]

    result = _enumerate_files(paths, EnumerateBy.MTIME, archive)

    assert [result[i].name for i in range(3)] == ["file_a.txt", "file_b.txt", "file_c.txt"]


def test_glob_matrix_stream_enumerate_by_name_with_arbitrary_filenames(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, n = arbitrary_named_txt_matrices
    stream = GlobMatrixStream(
        str(mat_dir / "E1_*_matrix.txt"),
        enumerate_by=EnumerateBy.NAME,
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)
    sample = stream.load_dense_sample(0)
    assert sample.matrix.shape == (n, n)


def test_glob_vector_stream_enumerate_by_name_with_arbitrary_filenames(
    archive: InMemoryArchiveSource, readers: MatrixReaders, arbitrary_named_txt_vectors: Path
) -> None:
    stream = GlobVectorStream(
        str(arbitrary_named_txt_vectors / "E1_*_rhs.txt"),
        enumerate_by=EnumerateBy.NAME,
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)
    assert stream.load_sample(0).vector.shape == (5,)


def test_open_matrix_stream_glob_with_enumerate_by_name(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, _ = arbitrary_named_txt_matrices
    stream = open_matrix_stream(
        str(mat_dir / "E1_*_matrix.txt"),
        enumerate_by=EnumerateBy.NAME,
        readers=readers,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)


def test_glob_matrix_stream_enumerate_by_name_is_reproducible(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, _ = arbitrary_named_txt_matrices
    expr = str(mat_dir / "E1_*_matrix.txt")

    stream1 = GlobMatrixStream(
        expr, enumerate_by=EnumerateBy.NAME, reader=readers.generic, archive_source=archive
    )
    stream2 = GlobMatrixStream(
        expr, enumerate_by=EnumerateBy.NAME, reader=readers.generic, archive_source=archive
    )

    assert stream1.sample_ids == stream2.sample_ids
    for sid in stream1.sample_ids:
        np.testing.assert_array_equal(
            stream1.load_dense_sample(sid).matrix,
            stream2.load_dense_sample(sid).matrix,
        )


def test_source_config_rejects_both_sample_id_regex_and_enumerate_by() -> None:
    from neuralls.platform.config.models.data_models import SourceConfig

    with pytest.raises(ValidationError):
        SourceConfig(sample_id_regex=r"(\d+)", enumerate_by=EnumerateBy.NAME)


def test_source_config_accepts_enumerate_by_without_regex() -> None:
    from neuralls.platform.config.models.data_models import SourceConfig

    config = SourceConfig(enumerate_by=EnumerateBy.NAME)
    assert config.enumerate_by == EnumerateBy.NAME


def test_glob_matrix_stream_exclude_indices_drops_sample_without_renumbering(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    stream = GlobMatrixStream(
        str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
        enumerate_by=EnumerateBy.NAME,
        exclude_indices=(1,),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 2)


def test_glob_matrix_stream_include_indices_restricts_to_subset(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    stream = GlobMatrixStream(
        str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
        enumerate_by=EnumerateBy.NAME,
        include_indices=(0, 2),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 2)


def test_glob_vector_stream_exclude_indices_drops_sample(
    archive: InMemoryArchiveSource, readers: MatrixReaders, arbitrary_named_txt_vectors: Path
) -> None:
    stream = GlobVectorStream(
        str(arbitrary_named_txt_vectors / "E1_*_rhs.txt"),
        enumerate_by=EnumerateBy.NAME,
        exclude_indices=(0,),
        reader=readers.generic,
        archive_source=archive,
    )

    assert stream.sample_ids == (1, 2)


def test_glob_matrix_stream_include_and_exclude_indices_are_mutually_exclusive(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            include_indices=(0,),
            exclude_indices=(1,),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_matrix_stream_include_indices_rejects_unknown_id(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            include_indices=(0, 99),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_matrix_stream_exclude_indices_rejects_unknown_id(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    """A typo'd exclude id must fail loudly, not silently leave the leak in place."""
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            exclude_indices=(99,),
            reader=readers.generic,
            archive_source=archive,
        )


def test_glob_matrix_stream_exclude_indices_rejects_emptying_all_samples(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    in_memory_subdomain_matrices: Path,
) -> None:
    with pytest.raises(ValueError):
        GlobMatrixStream(
            str(in_memory_subdomain_matrices / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            exclude_indices=(0, 1, 2),
            reader=readers.generic,
            archive_source=archive,
        )


def test_open_matrix_stream_exclude_indices_requires_glob_source(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    matrix_path = archive.add(ARCHIVE_DIR / "matrix.npy", np.eye(3))

    with pytest.raises(ValueError):
        open_matrix_stream(
            str(matrix_path), exclude_indices=(0,), readers=readers, archive_source=archive
        )


def test_source_config_rejects_both_include_and_exclude_indices() -> None:
    from neuralls.platform.config.models.data_models import SourceConfig

    with pytest.raises(ValidationError):
        SourceConfig(include_indices=(0, 1), exclude_indices=(2,))


def test_bind_sources_pairs_solution_ids_per_matrix() -> None:
    """solution_ids are bound per-matrix, mirroring the rhs_ids pattern."""
    bindings = bind_sources(
        matrix_ids=(0, 1, 2),
        solution_ids=(0, 1, 2),
    )
    assert [b.matrix_sample_id for b in bindings] == [0, 1, 2]
    assert [b.solution_sample_id for b in bindings] == [0, 1, 2]


def test_bind_sources_solution_mismatch_raises() -> None:
    """Mismatched solution_ids vs matrix_ids raises ValueError for multi-matrix case."""
    with pytest.raises(ValueError):
        bind_sources(
            matrix_ids=(0, 1, 2),
            solution_ids=(0, 1),  # missing id 2
        )


def test_bind_sources_single_matrix_broadcasts_to_many_solution_ids() -> None:
    """Single matrix broadcasts across solution_ids (mirrors rhs_ids broadcast)."""
    bindings = bind_sources(
        matrix_ids=(0,),
        solution_ids=(0, 1, 2),
    )
    assert len(bindings) == 3
    assert all(b.matrix_sample_id == 0 for b in bindings)
    assert [b.solution_sample_id for b in bindings] == [0, 1, 2]
    assert all(b.rhs_sample_id is None for b in bindings)


def test_bind_sources_parameter_ids_must_match_matrix_ids() -> None:
    """A parameter stream with an id no matrix has is rejected, not silently unbound."""
    with pytest.raises(ValueError):
        bind_sources(
            matrix_ids=(0, 1),
            parameters_ids_list=((0, 1, 2),),
        )


def test_bind_sources_single_matrix_rejects_extra_parameter_ids() -> None:
    """One matrix with several parameter ids is an input error, not a broadcast."""
    with pytest.raises(ValueError):
        bind_sources(
            matrix_ids=(0,),
            rhs_ids=(0,),
            parameters_ids_list=((0, 1, 2),),
        )


def test_bind_sources_each_binding_takes_its_matrix_parameter() -> None:
    """Parameters belong to the matrix, so every rhs of one matrix reuses that matrix's id."""
    bindings = bind_sources(
        matrix_ids=(0,),
        rhs_ids=(0, 1, 2),
        parameters_ids_list=((0,),),
    )
    assert [b.parameters_sample_ids for b in bindings] == [(0,), (0,), (0,)]
