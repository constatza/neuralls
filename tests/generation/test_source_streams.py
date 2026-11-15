"""Matrix and vector source streams over an in-memory archive.

Every stream rule is tested once per stream kind (matrix and vector twins share a table).
The sample ids a stream derives, the files a glob matches, the index filters and the
per-source shape checks belong to ``source_streams``; ``bind_sources`` pairs the resulting ids.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
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
    NpyMatrixStream,
    NpyVectorStream,
    TxtMatrixStream,
    TxtVectorStream,
    open_matrix_stream,
    open_vector_stream,
)
from neuralls.platform.config.models.data_models import SourceConfig
from neuralls.platform.storage.archive_source import FsArchiveSource
from neuralls.platform.storage.matrix_readers import read_matrix
from tests.fakes.archive_source import InMemoryArchiveSource

ARCHIVE_DIR = Path("archive")

# Stems end in non-contiguous, unsorted integer ids; sorted by name they read 007, 10, 2.
NUMBERED_MARKERS = {"2": 2.0, "10": 10.0, "007": 7.0}


def _marker_matrix(value: float) -> np.ndarray:
    """A 2x2 matrix whose contents identify one source file."""
    return np.array([[value, 0.0], [0.0, value + 1.0]], dtype=np.float64)


def _marker_vector(value: float) -> np.ndarray:
    """A length-3 vector whose contents identify one source file."""
    return np.array([value, value + 1.0, value + 2.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StreamKind:
    """What differs between the matrix and the vector stream family.

    Attributes:
        name: Subdirectory of the archive holding this kind's files.
        stem: Filename prefix (``A_2.txt`` or ``b_2.txt``).
        marker: Array that identifies one source file by ``value``.
        glob_stream: Glob stream class.
        npy_stream: Single-``.npy`` stream class.
        txt_stream: Single-``.txt`` stream class.
        open: Factory choosing the stream class by path expression.
        load: Array of one loaded sample.
        iterations: Sample ids of every iteration method the stream offers.
        npy_stack: Array of two samples stacked in one ``.npy`` file.
        npy_too_many_dims: ``.npy`` array whose rank no stream of this kind accepts.
        txt_wrong_rank: ``.txt`` array whose rank this kind rejects.
    """

    name: str
    stem: str
    marker: Callable[[float], np.ndarray]
    glob_stream: type
    npy_stream: type
    txt_stream: type
    open: Callable[..., Any]
    load: Callable[[Any, int], np.ndarray]
    iterations: Callable[[Any], list[list[int]]]
    npy_stack: np.ndarray
    npy_too_many_dims: np.ndarray
    txt_wrong_rank: np.ndarray

    def single(self, suffix: str, path: Path, readers: MatrixReaders) -> Any:
        """The single-file stream for ``suffix`` (``npy`` or ``txt``)."""
        if suffix == "npy":
            return self.npy_stream(path, reader=readers.dense)
        return self.txt_stream(path, reader=readers.generic)

    def glob(
        self, expression: str, archive: InMemoryArchiveSource, readers: MatrixReaders, **kwargs: Any
    ) -> Any:
        """A glob stream over ``archive`` with the generic reader."""
        return self.glob_stream(
            expression, reader=readers.generic, archive_source=archive, **kwargs
        )

    def add_files(
        self, archive: InMemoryArchiveSource, directory: Path, names: dict[str, float]
    ) -> Path:
        """Register one marker file per ``name`` under ``directory`` and return it."""
        for name, value in names.items():
            archive.add(directory / name, self.marker(value))
        return directory


MATRIX_KIND = StreamKind(
    name="matrices",
    stem="A",
    marker=_marker_matrix,
    glob_stream=GlobMatrixStream,
    npy_stream=NpyMatrixStream,
    txt_stream=TxtMatrixStream,
    open=open_matrix_stream,
    load=lambda stream, sample_id: stream.load_dense_sample(sample_id).matrix,
    iterations=lambda stream: [
        [s.sample_id for s in stream.iter_dense_samples()],
        [s.sample_id for s in stream.iter_sparse_samples()],
    ],
    npy_stack=np.stack([_marker_matrix(1.0), _marker_matrix(5.0)]),
    npy_too_many_dims=np.ones((2, 2, 2, 2), dtype=np.float64),
    txt_wrong_rank=np.array([1.0, 2.0, 3.0]),
)
VECTOR_KIND = StreamKind(
    name="vectors",
    stem="b",
    marker=_marker_vector,
    glob_stream=GlobVectorStream,
    npy_stream=NpyVectorStream,
    txt_stream=TxtVectorStream,
    open=open_vector_stream,
    load=lambda stream, sample_id: stream.load_sample(sample_id).vector,
    iterations=lambda stream: [[s.sample_id for s in stream.iter_samples()]],
    npy_stack=np.stack([_marker_vector(1.0), _marker_vector(5.0)]),
    npy_too_many_dims=np.ones((2, 2, 2), dtype=np.float64),
    txt_wrong_rank=np.ones((3, 2), dtype=np.float64),
)


@pytest.fixture(params=[MATRIX_KIND, VECTOR_KIND], ids=["matrix", "vector"])
def kind(request: pytest.FixtureRequest) -> StreamKind:
    """Each stream family in turn."""
    return request.param


@pytest.fixture
def archive() -> InMemoryArchiveSource:
    """An empty in-memory archive; each test registers the files it reads."""
    return InMemoryArchiveSource()


@pytest.fixture
def readers(archive: InMemoryArchiveSource) -> MatrixReaders:
    """Readers over the arrays registered in ``archive``."""
    return archive.readers()


@pytest.fixture
def numbered_dir(kind: StreamKind, archive: InMemoryArchiveSource) -> Path:
    """Files ``<stem>_2``, ``<stem>_10``, ``<stem>_007`` (``.txt``) with the markers above."""
    return kind.add_files(
        archive,
        ARCHIVE_DIR / f"numbered_{kind.name}",
        {f"{kind.stem}_{digits}.txt": value for digits, value in NUMBERED_MARKERS.items()},
    )


# --- sample ids and iteration ----------------------------------------------------


def test_glob_derives_ids_from_trailing_digits_and_iterates_in_id_order(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, numbered_dir: Path
) -> None:
    stream = kind.glob(str(numbered_dir / f"{kind.stem}_*.txt"), archive, readers)

    assert stream.sample_ids == (2, 7, 10)
    assert {tuple(ids) for ids in kind.iterations(stream)} == {(2, 7, 10)}
    for sample_id, value in ((2, 2.0), (7, 7.0), (10, 10.0)):
        np.testing.assert_array_equal(kind.load(stream, sample_id), kind.marker(value))


def test_glob_custom_regex_selects_leading_digits(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    directory = kind.add_files(
        archive, ARCHIVE_DIR / "prefixed", {"12_case_3.txt": 12.0, "40_case_3.txt": 40.0}
    )

    stream = kind.glob(str(directory / "*_case_3.txt"), archive, readers, sample_id_regex=r"(\d+)")

    assert stream.sample_ids == (12, 40)
    np.testing.assert_array_equal(kind.load(stream, 12), kind.marker(12.0))


@pytest.mark.parametrize(
    ("names", "pattern"),
    [
        pytest.param(("alpha.txt",), "*.txt", id="filename-without-digits"),
        pytest.param(("a_1.txt", "b_1.txt"), "*_1.txt", id="duplicate-sample-ids"),
    ],
)
def test_glob_rejects_unusable_sample_ids(
    kind: StreamKind,
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    names: tuple[str, ...],
    pattern: str,
) -> None:
    directory = kind.add_files(
        archive, ARCHIVE_DIR / "bad_ids", {name: float(i) for i, name in enumerate(names)}
    )

    with pytest.raises(ValueError):
        kind.glob(str(directory / pattern), archive, readers)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("absent/{stem}_*.txt", id="missing-parent-directory"),
        pytest.param("numbered_{name}/z_*.txt", id="no-match"),
    ],
)
def test_glob_without_files_raises(
    kind: StreamKind,
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    numbered_dir: Path,
    expression: str,
) -> None:
    pattern = ARCHIVE_DIR / expression.format(stem=kind.stem, name=kind.name)

    with pytest.raises(FileNotFoundError):
        kind.glob(str(pattern), archive, readers)


def test_glob_enumerate_by_name_ignores_the_ids_in_filenames(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, numbered_dir: Path
) -> None:
    stream = kind.glob(
        str(numbered_dir / f"{kind.stem}_*.txt"), archive, readers, enumerate_by=EnumerateBy.NAME
    )

    # Sorted names are 007, 10, 2.
    assert stream.sample_ids == (0, 1, 2)
    for sample_id, value in ((0, 7.0), (1, 10.0), (2, 2.0)):
        np.testing.assert_array_equal(kind.load(stream, sample_id), kind.marker(value))


def test_factory_passes_enumerate_by_to_the_glob(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, numbered_dir: Path
) -> None:
    stream = kind.open(
        str(numbered_dir / f"{kind.stem}_*.txt"),
        enumerate_by=EnumerateBy.NAME,
        readers=readers,
        archive_source=archive,
    )

    assert stream.sample_ids == (0, 1, 2)


def test_glob_unknown_sample_id_raises_key_error(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, numbered_dir: Path
) -> None:
    stream = kind.glob(str(numbered_dir / f"{kind.stem}_*.txt"), archive, readers)

    with pytest.raises(KeyError):
        kind.load(stream, 99)


# --- index filters ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "loaded_markers"),
    [
        pytest.param({"include_indices": (10, 2)}, {2: 2.0, 10: 10.0}, id="include"),
        pytest.param({"exclude_indices": (7,)}, {2: 2.0, 10: 10.0}, id="exclude"),
        pytest.param(
            {"enumerate_by": EnumerateBy.NAME, "include_indices": (0, 2)},
            {0: 7.0, 2: 2.0},
            id="include-by-name",
        ),
        pytest.param(
            {"enumerate_by": EnumerateBy.NAME, "exclude_indices": (1,)},
            {0: 7.0, 2: 2.0},
            id="exclude-by-name",
        ),
    ],
)
def test_glob_index_filters_keep_the_ids_of_the_surviving_files(
    kind: StreamKind,
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    numbered_dir: Path,
    kwargs: dict[str, Any],
    loaded_markers: dict[int, float],
) -> None:
    stream = kind.glob(str(numbered_dir / f"{kind.stem}_*.txt"), archive, readers, **kwargs)

    assert stream.sample_ids == tuple(loaded_markers)
    for sample_id, value in loaded_markers.items():
        np.testing.assert_array_equal(kind.load(stream, sample_id), kind.marker(value))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"include_indices": (2,), "exclude_indices": (7,)}, id="include-and-exclude"),
        pytest.param({"include_indices": (2, 99)}, id="include-unknown-id"),
        pytest.param({"exclude_indices": (99,)}, id="exclude-unknown-id"),
        pytest.param({"exclude_indices": (2, 7, 10)}, id="exclude-every-sample"),
    ],
)
def test_glob_rejects_inconsistent_index_filters(
    kind: StreamKind,
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    numbered_dir: Path,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        kind.glob(str(numbered_dir / f"{kind.stem}_*.txt"), archive, readers, **kwargs)


@pytest.mark.parametrize("option", ["include_indices", "exclude_indices"])
def test_factory_index_filters_require_a_glob_source(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, option: str
) -> None:
    path = archive.add(ARCHIVE_DIR / "single.npy", kind.marker(1.0))

    with pytest.raises(ValueError):
        kind.open(str(path), readers=readers, archive_source=archive, **{option: (0,)})


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"sample_id_regex": r"(\d+)", "enumerate_by": EnumerateBy.NAME}, id="regex"),
        pytest.param({"include_indices": (0, 1), "exclude_indices": (2,)}, id="indices"),
    ],
)
def test_source_config_rejects_conflicting_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SourceConfig(**kwargs)


# --- glob loading ----------------------------------------------------------------


def test_glob_loads_npy_and_txt_from_one_pattern(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    directory = ARCHIVE_DIR / "mixed"
    archive.add(directory / f"{kind.stem}_1.txt", kind.marker(1.0))
    archive.add(directory / f"{kind.stem}_2.npy", kind.marker(2.0))

    stream = kind.glob(str(directory / f"{kind.stem}_*"), archive, readers)

    assert stream.sample_ids == (1, 2)
    np.testing.assert_array_equal(kind.load(stream, 1), kind.marker(1.0))
    np.testing.assert_array_equal(kind.load(stream, 2), kind.marker(2.0))


def test_glob_rejects_unsupported_extension_at_load(
    kind: StreamKind, archive: InMemoryArchiveSource
) -> None:
    """The suffix check is the real reader's; it fails before reading any bytes."""
    archive.add(ARCHIVE_DIR / "csv" / f"{kind.stem}_1.csv", kind.marker(1.0))
    stream = kind.glob_stream(
        str(ARCHIVE_DIR / "csv" / f"{kind.stem}_*"),
        reader=partial(read_matrix, lazy=True),
        archive_source=archive,
    )

    with pytest.raises(ValueError):
        kind.load(stream, 1)


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        pytest.param(
            "A_1.npy", np.stack([_marker_matrix(1.0), _marker_matrix(2.0)]), id="stacked-npy"
        ),
        pytest.param("A_1.txt", np.array([1.0, 2.0, 3.0]), id="flat-txt"),
    ],
)
def test_glob_matrix_rejects_non_2d_file(
    archive: InMemoryArchiveSource, readers: MatrixReaders, name: str, contents: np.ndarray
) -> None:
    archive.add(ARCHIVE_DIR / "bad" / name, contents)
    stream = MATRIX_KIND.glob(str(ARCHIVE_DIR / "bad" / "A_*"), archive, readers)

    with pytest.raises(ValueError):
        MATRIX_KIND.load(stream, 1)


def test_glob_vector_reshapes_a_column_npy_to_one_vector(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    """A glob-matched (n,1) .npy is one column vector, not a stack of n scalars."""
    archive.add(ARCHIVE_DIR / "columns" / "b_1.npy", _marker_vector(1.0).reshape(3, 1))

    stream = VECTOR_KIND.glob(str(ARCHIVE_DIR / "columns" / "b_*.npy"), archive, readers)

    assert stream.sample_ids == (1,)
    np.testing.assert_array_equal(VECTOR_KIND.load(stream, 1), _marker_vector(1.0))


def test_glob_vector_rejects_a_genuinely_2d_npy(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    archive.add(ARCHIVE_DIR / "wide" / "b_1.npy", np.ones((3, 2), dtype=np.float64))
    stream = VECTOR_KIND.glob(str(ARCHIVE_DIR / "wide" / "b_*.npy"), archive, readers)

    with pytest.raises(ValueError):
        VECTOR_KIND.load(stream, 1)


@pytest.mark.parametrize(
    ("path", "expression", "contents", "sample_id", "dense"),
    [
        pytest.param(
            ARCHIVE_DIR / "stack.npy",
            str(ARCHIVE_DIR / "stack.npy"),
            np.stack([_marker_matrix(2.0), _marker_matrix(4.0)]),
            1,
            _marker_matrix(4.0),
            id="npy-stack",
        ),
        pytest.param(
            ARCHIVE_DIR / "sparse_source" / "A_1.txt",
            str(ARCHIVE_DIR / "sparse_source" / "A_*.txt"),
            np.array([[0.0, 2.0], [3.0, 0.0]]),
            1,
            np.array([[0.0, 2.0], [3.0, 0.0]]),
            id="glob-txt",
        ),
    ],
)
def test_sparse_sample_components_reconstruct_the_dense_matrix(
    archive: InMemoryArchiveSource,
    readers: MatrixReaders,
    path: Path,
    expression: str,
    contents: np.ndarray,
    sample_id: int,
    dense: np.ndarray,
) -> None:
    archive.add(path, contents)

    sample = open_matrix_stream(
        expression, readers=readers, archive_source=archive
    ).load_sparse_sample(sample_id)

    reconstructed = np.zeros(sample.size, dtype=np.float64)
    reconstructed[sample.indices[0], sample.indices[1]] = sample.values
    assert sample.size == dense.shape
    np.testing.assert_array_equal(reconstructed, dense)


# --- single-file streams ---------------------------------------------------------


@pytest.mark.parametrize("suffix", ["npy", "txt"])
def test_single_file_stream_exposes_one_sample(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, suffix: str
) -> None:
    path = archive.add(ARCHIVE_DIR / f"single.{suffix}", kind.marker(1.0))

    stream = kind.single(suffix, path, readers)

    assert stream.sample_ids == (0,)
    assert {tuple(ids) for ids in kind.iterations(stream)} == {(0,)}
    np.testing.assert_array_equal(kind.load(stream, 0), kind.marker(1.0))
    with pytest.raises(KeyError):
        kind.load(stream, 1)


@pytest.mark.parametrize("suffix", ["npy", "txt"])
def test_single_file_stream_rejects_an_array_of_the_wrong_rank(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders, suffix: str
) -> None:
    bad = kind.npy_too_many_dims if suffix == "npy" else kind.txt_wrong_rank
    path = archive.add(ARCHIVE_DIR / f"bad.{suffix}", bad)

    with pytest.raises(ValueError):
        kind.load(kind.single(suffix, path, readers), 0)


def test_factory_npy_stack_exposes_every_sample(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "stack.npy", kind.npy_stack)

    stream = kind.open(str(path), readers=readers, archive_source=archive)

    assert stream.sample_ids == (0, 1)
    assert {tuple(ids) for ids in kind.iterations(stream)} == {(0, 1)}
    np.testing.assert_array_equal(kind.load(stream, 1), kind.npy_stack[1])


def test_single_file_vector_npy_treats_a_column_as_a_stack_of_scalars(
    archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    """Unlike the glob variant, a single-file (n,1) .npy is n length-1 vectors."""
    path = archive.add(ARCHIVE_DIR / "column.npy", _marker_vector(1.0).reshape(3, 1))

    stream = NpyVectorStream(path, reader=readers.dense)

    assert stream.sample_ids == (0, 1, 2)
    np.testing.assert_array_equal(VECTOR_KIND.load(stream, 0), np.array([1.0]))


# --- factory dispatch ------------------------------------------------------------


def test_factory_selects_the_stream_type_by_expression(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    npy_path = archive.add(ARCHIVE_DIR / "one.npy", kind.marker(1.0))
    txt_path = archive.add(ARCHIVE_DIR / "one.txt", kind.marker(1.0))
    archive.add(ARCHIVE_DIR / "globbed" / f"{kind.stem}_1.txt", kind.marker(1.0))
    expressions = [
        str(npy_path),
        str(txt_path),
        str(ARCHIVE_DIR / "globbed" / f"{kind.stem}_*.txt"),
    ]

    streams = [kind.open(e, readers=readers, archive_source=archive) for e in expressions]

    assert [type(stream) for stream in streams] == [
        kind.npy_stream,
        kind.txt_stream,
        kind.glob_stream,
    ]


def test_factory_rejects_unknown_suffix(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    path = archive.add(ARCHIVE_DIR / "single.csv", kind.marker(1.0))

    with pytest.raises(ValueError):
        kind.open(str(path), readers=readers, archive_source=archive)


def test_factory_missing_file_raises(
    kind: StreamKind, archive: InMemoryArchiveSource, readers: MatrixReaders
) -> None:
    with pytest.raises(FileNotFoundError):
        kind.open(str(ARCHIVE_DIR / "absent.npy"), readers=readers, archive_source=archive)


# --- enumeration order -----------------------------------------------------------


@pytest.fixture
def staggered_files(archive: InMemoryArchiveSource) -> list[Path]:
    """Files c, a, b created in that order, each later than the last, in both clocks."""
    return [
        archive.add(ARCHIVE_DIR / f"file_{suffix}.txt", np.zeros(1), mtime=float(i), ctime=float(i))
        for i, suffix in enumerate(["c", "a", "b"])
    ]


def test_enumerate_by_name_assigns_ids_in_name_order(archive: InMemoryArchiveSource) -> None:
    paths = [ARCHIVE_DIR / f"file_z_{suffix}.txt" for suffix in ["c", "a", "b"]]

    result = _enumerate_files(paths, EnumerateBy.NAME, archive)

    assert {i: path.name for i, path in result.items()} == {
        0: "file_z_a.txt",
        1: "file_z_b.txt",
        2: "file_z_c.txt",
    }


@pytest.mark.parametrize("by", [EnumerateBy.MTIME, EnumerateBy.CTIME])
def test_enumerate_by_time_assigns_ids_in_creation_order(
    archive: InMemoryArchiveSource, staggered_files: list[Path], by: EnumerateBy
) -> None:
    result = _enumerate_files(staggered_files, by, archive)

    assert {i: path.name for i, path in result.items()} == {
        0: "file_c.txt",
        1: "file_a.txt",
        2: "file_b.txt",
    }


def test_enumerate_by_ctime_raises_without_creation_times(archive: InMemoryArchiveSource) -> None:
    paths = [archive.add(ARCHIVE_DIR / name, np.zeros(1), mtime=0.0) for name in ("a.txt", "b.txt")]

    with pytest.raises(ValueError, match="creation times"):
        _enumerate_files(paths, EnumerateBy.CTIME, archive)


def test_enumerate_breaks_timestamp_ties_by_name(archive: InMemoryArchiveSource) -> None:
    paths = [
        archive.add(ARCHIVE_DIR / f"file_{suffix}.txt", np.zeros(1), mtime=1_000_000.0)
        for suffix in ["c", "b", "a"]
    ]

    result = _enumerate_files(paths, EnumerateBy.MTIME, archive)

    assert {i: path.name for i, path in result.items()} == {
        0: "file_a.txt",
        1: "file_b.txt",
        2: "file_c.txt",
    }


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows stat results always carry st_birthtime"
)
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


# --- binding ids -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"matrix_ids": (0,), "rhs_ids": (0, 1, 2)},
            [(0, 0, None, ()), (0, 1, None, ()), (0, 2, None, ())],
            id="single-matrix-broadcasts-to-rhs",
        ),
        pytest.param(
            {"matrix_ids": (0, 1, 2), "solution_ids": (0, 1, 2)},
            [(0, None, 0, ()), (1, None, 1, ()), (2, None, 2, ())],
            id="solutions-pair-per-matrix",
        ),
        pytest.param(
            {"matrix_ids": (0,), "solution_ids": (0, 1, 2)},
            [(0, None, 0, ()), (0, None, 1, ()), (0, None, 2, ())],
            id="single-matrix-broadcasts-to-solutions",
        ),
        pytest.param(
            {"matrix_ids": (0,), "rhs_ids": (0, 1, 2), "parameters_ids_list": ((0,),)},
            [(0, 0, None, (0,)), (0, 1, None, (0,)), (0, 2, None, (0,))],
            id="every-rhs-of-a-matrix-reuses-its-parameter",
        ),
    ],
)
def test_bind_sources_pairs_ids(
    kwargs: dict[str, Any], expected: list[tuple[int, int | None, int | None, tuple[int, ...]]]
) -> None:
    bindings = bind_sources(**kwargs)

    assert [
        (b.matrix_sample_id, b.rhs_sample_id, b.solution_sample_id, b.parameters_sample_ids)
        for b in bindings
    ] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"matrix_ids": (0, 1, 2), "solution_ids": (0, 1)}, id="solution-count"),
        pytest.param(
            {"matrix_ids": (0, 1), "parameters_ids_list": ((0, 1, 2),)}, id="parameter-id-no-matrix"
        ),
        pytest.param(
            {"matrix_ids": (0,), "rhs_ids": (0,), "parameters_ids_list": ((0, 1, 2),)},
            id="single-matrix-extra-parameters",
        ),
    ],
)
def test_bind_sources_rejects_mismatched_ids(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        bind_sources(**kwargs)
