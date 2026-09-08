"""Streaming source readers for matrix and vector sample ingestion.

This module keeps source discovery and per-sample loading decoupled from
generation strategy logic. It supports:
- single .txt matrix/vector files
- single .npy matrix/vector files (including stacked samples via mmap)
- glob patterns of .txt/.npy files (one sample per file)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_GLOB_CHARS = ("*", "?", "[", "]")
_DEFAULT_SAMPLE_ID_REGEX = r"(\d+)(?!.*\d)"


class EnumerateBy(StrEnum):
    """Criterion for assigning sequential IDs to glob-matched files.

    Use when filenames carry no natural integer ID (e.g. parameter-encoded names
    like ``E1_3000_E2_78000_matrix.txt``).  Files are sorted by the chosen
    criterion and assigned sequential IDs 0, 1, 2, …
    """

    NAME = "name"
    CTIME = "ctime"
    MTIME = "mtime"


def _sort_key_for(path: Path, by: EnumerateBy) -> float | str:
    """Return the sort key for *path* under the given *by* strategy."""
    match by:
        case EnumerateBy.NAME:
            return path.name
        case EnumerateBy.CTIME:
            return path.stat().st_ctime
        case EnumerateBy.MTIME:
            return path.stat().st_mtime


def _enumerate_files(paths: Sequence[Path], by: EnumerateBy) -> dict[int, Path]:
    """Assign sequential IDs to *paths* sorted by *by*, returning ``{id: path}``."""
    return {i: p for i, p in enumerate(sorted(paths, key=lambda p: _sort_key_for(p, by)))}


def _filter_mapping(
    mapping: dict[int, Path],
    *,
    include_indices: tuple[int, ...] | None,
    exclude_indices: tuple[int, ...],
) -> dict[int, Path]:
    """Restrict *mapping* to `include_indices`, or drop `exclude_indices`.

    Keeps original sample ids as dict keys (no renumbering) so downstream
    keyed lookups (`bind_sources`, `load_dense_sample`) stay correct.

    # ponytail: hand-maintained id lists per dataset TOML are a crude,
    # manual train/eval split — refine into a shared, seeded split utility
    # (e.g. fractional or stratified) if more parametric-family cases need
    # this, so the held-out ids aren't hardcoded and duplicated per config.
    """
    if include_indices is not None and exclude_indices:
        raise ValueError("include_indices and exclude_indices are mutually exclusive.")
    if include_indices is not None:
        missing = set(include_indices) - mapping.keys()
        if missing:
            raise ValueError(f"include_indices references unknown sample ids: {sorted(missing)}")
        return {i: mapping[i] for i in include_indices}
    if exclude_indices:
        missing = set(exclude_indices) - mapping.keys()
        if missing:
            raise ValueError(f"exclude_indices references unknown sample ids: {sorted(missing)}")
        return {i: p for i, p in mapping.items() if i not in exclude_indices}
    return mapping


def _is_glob_expression(expr: str) -> bool:
    """Return True when the path expression contains glob meta characters."""
    return any(char in expr for char in _GLOB_CHARS)


def _normalize_vector(array: np.ndarray, source: Path) -> np.ndarray:
    """Normalize vector shape to 1D float64."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.reshape(-1)
    raise ValueError(f"Expected vector from {source}, got shape {arr.shape}")


def _extract_sample_id(path: Path, pattern: re.Pattern[str]) -> int:
    """Extract integer sample id from filename stem."""
    match = pattern.search(path.stem)
    if match is None:
        raise ValueError(
            f"Could not extract sample id from '{path.name}' using regex '{pattern.pattern}'."
        )
    return int(match.group(1))


def _dense_to_sparse_components(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert dense matrix to COO payload components."""
    rows, cols = np.nonzero(matrix)
    values = np.asarray(matrix[rows, cols], dtype=np.float64)
    indices = np.vstack((rows, cols)).astype(np.int64, copy=False)
    return indices, values, (int(matrix.shape[0]), int(matrix.shape[1]))


@dataclass(frozen=True)
class DenseMatrixSample:
    """One dense matrix sample."""

    sample_id: int
    matrix: np.ndarray


@dataclass(frozen=True)
class SparseMatrixSample:
    """One sparse matrix sample as COO payload arrays."""

    sample_id: int
    indices: np.ndarray
    values: np.ndarray
    size: tuple[int, int]


@dataclass(frozen=True)
class VectorSample:
    """One RHS/solution vector sample."""

    sample_id: int
    vector: np.ndarray


@dataclass(frozen=True)
class SystemBinding:
    """ID-level binding across matrix/rhs/parameters/solution sources."""

    sample_id: int
    matrix_sample_id: int
    rhs_sample_id: int | None = None
    parameters_sample_ids: tuple[int | None, ...] = ()
    solution_sample_id: int | None = None


@runtime_checkable
class MatrixSampleStream(Protocol):
    """Protocol for lazily loading matrix samples."""

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        """Load one matrix sample in dense float64 format."""
        ...

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        """Load one matrix sample as sparse COO components."""
        ...

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        """Iterate all dense matrix samples."""
        ...

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        """Iterate all sparse matrix samples."""
        ...


@runtime_checkable
class VectorSampleStream(Protocol):
    """Protocol for lazily loading vector samples."""

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def load_sample(self, sample_id: int) -> VectorSample:
        """Load one vector sample."""
        ...

    def iter_samples(self) -> Iterator[VectorSample]:
        """Iterate all vector samples."""
        ...


@dataclass(frozen=True)
class _RawSample:
    """One sample's array exactly as read from disk, plus the file it came from.

    Attributes:
        sample_id: Sample id this array belongs to.
        array: Raw array (possibly memory-mapped, possibly not yet float64).
        origin: File the array was read from, used for error messages.
    """

    sample_id: int
    array: np.ndarray
    origin: Path


class _RawSampleSource(Protocol):
    """Protocol for locating sample files and reading their raw arrays.

    A source owns *where* samples come from (one stacked .npy, one .txt, or a
    glob of per-sample files); the stream wrapped around it owns *what* a sample
    means (dense matrix vs 1D vector).
    """

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def read(self, sample_id: int) -> _RawSample:
        """Read one sample's raw array."""
        ...


def _index_by_sample_id_regex(
    paths: Sequence[Path],
    *,
    noun: str,
    sample_id_regex: str | None,
) -> dict[int, Path]:
    """Map filename-derived sample ids to *paths*, rejecting duplicate ids."""
    regex = re.compile(sample_id_regex or _DEFAULT_SAMPLE_ID_REGEX)
    mapping: dict[int, Path] = {}
    for path in paths:
        sample_id = _extract_sample_id(path, regex)
        if sample_id in mapping:
            raise ValueError(
                f"Duplicate {noun} sample id {sample_id} for files {mapping[sample_id]} and {path}"
            )
        mapping[sample_id] = path
    return mapping


def _build_glob_index(
    expr: str,
    *,
    noun: str,
    sample_id_regex: str | None,
    enumerate_by: EnumerateBy | None,
    include_indices: tuple[int, ...] | None,
    exclude_indices: tuple[int, ...],
) -> dict[int, Path]:
    """Resolve a glob expression to ``{sample_id: path}``.

    Args:
        expr: Glob expression whose parent directory must already exist.
        noun: Source kind used in error messages ("matrix" or "vector").
        sample_id_regex: Regex whose first group holds the id in the file stem.
            Ignored when *enumerate_by* is given; defaults to the trailing
            integer of the stem.
        enumerate_by: Assign sequential ids by this criterion instead of parsing
            them out of the filenames.
        include_indices: Keep only these sample ids, if given.
        exclude_indices: Drop these sample ids.

    Returns:
        Mapping of sample id to source file, without renumbering.
    """
    pattern_path = Path(expr)
    parent = pattern_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"{noun.capitalize()} glob parent directory not found: {parent}")
    paths = sorted(parent.glob(pattern_path.name))
    if not paths:
        raise FileNotFoundError(f"No {noun} files match glob: {expr}")
    mapping = (
        _enumerate_files(paths, enumerate_by)
        if enumerate_by is not None
        else _index_by_sample_id_regex(paths, noun=noun, sample_id_regex=sample_id_regex)
    )
    mapping = _filter_mapping(
        mapping, include_indices=include_indices, exclude_indices=exclude_indices
    )
    if not mapping:
        raise ValueError(f"No {noun} samples remain after filtering glob: {expr}")
    return mapping


def _read_sample_file(path: Path, *, noun: str) -> np.ndarray:
    """Read one per-sample .npy (memory-mapped) or .txt file."""
    match path.suffix:
        case ".npy":
            return np.load(path, mmap_mode="r")
        case ".txt":
            return np.loadtxt(path, dtype=np.float64)
        case _:
            raise ValueError(f"Unsupported {noun} file extension in glob: {path.suffix}")


class _NpyFileSource:
    """Samples from one .npy file holding a single sample or a stack of them."""

    def __init__(self, path: Path, *, noun: str, sample_ndim: int, shape_error: str) -> None:
        """Open *path* and determine how many samples it holds.

        Args:
            path: Source .npy file, opened with mmap.
            noun: Source kind used in error messages ("matrix" or "vector").
            sample_ndim: Rank of a single sample; rank ``sample_ndim + 1`` is
                read as a stack of samples along the leading axis.
            shape_error: Message template, formatted with ``shape``, raised when
                the file's rank is neither of the two accepted ones.
        """
        self._path = path
        self._noun = noun
        self._sample_ndim = sample_ndim
        self._array = np.load(path, mmap_mode="r")
        if self._array.ndim == sample_ndim:
            self._sample_ids = (0,)
        elif self._array.ndim == sample_ndim + 1:
            self._sample_ids = tuple(range(int(self._array.shape[0])))
        else:
            raise ValueError(shape_error.format(shape=self._array.shape))

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def read(self, sample_id: int) -> _RawSample:
        if sample_id not in self._sample_ids:
            raise KeyError(f"Unknown {self._noun} sample id {sample_id} for {self._path}")
        stacked = self._array.ndim > self._sample_ndim
        array = self._array[sample_id] if stacked else self._array
        return _RawSample(sample_id=sample_id, array=array, origin=self._path)


class _TxtFileSource:
    """The single sample held by one .txt file."""

    def __init__(self, path: Path, *, noun: str) -> None:
        self._path = path
        self._noun = noun

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return (0,)

    def read(self, sample_id: int) -> _RawSample:
        if sample_id != 0:
            raise KeyError(f"Unknown {self._noun} sample id {sample_id} for {self._path}")
        array = np.loadtxt(self._path, dtype=np.float64)
        return _RawSample(sample_id=0, array=array, origin=self._path)


class _GlobFileSource:
    """Samples from glob-matched .txt/.npy files, one sample per file."""

    def __init__(
        self,
        expr: str,
        *,
        noun: str,
        sample_id_regex: str | None,
        enumerate_by: EnumerateBy | None,
        include_indices: tuple[int, ...] | None,
        exclude_indices: tuple[int, ...],
    ) -> None:
        self._noun = noun
        self._mapping = _build_glob_index(
            expr,
            noun=noun,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
        )
        self._sample_ids = tuple(sorted(self._mapping.keys()))

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def read(self, sample_id: int) -> _RawSample:
        path = self._mapping.get(sample_id)
        if path is None:
            raise KeyError(f"Unknown {self._noun} sample id {sample_id}")
        array = _read_sample_file(path, noun=self._noun)
        return _RawSample(sample_id=sample_id, array=array, origin=path)


class _SampleStream[T](ABC):
    """Typed sample stream over any raw source.

    Subclasses turn a `_RawSample` into the sample type ``T`` they expose; every
    source-specific concern (which files, which ids, how to read them) belongs to
    the injected `_RawSampleSource`.
    """

    def __init__(self, source: _RawSampleSource) -> None:
        self._source = source

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        return self._source.sample_ids

    @abstractmethod
    def _build(self, raw: _RawSample) -> T:
        """Validate and convert one raw array into a typed sample."""

    def _load(self, sample_id: int) -> T:
        """Read and build one sample."""
        return self._build(self._source.read(sample_id))

    def _iter(self) -> Iterator[T]:
        """Iterate every sample in sample-id order."""
        for sample_id in self.sample_ids:
            yield self._load(sample_id)


class _MatrixStream(_SampleStream[DenseMatrixSample]):
    """`MatrixSampleStream` implementation over any raw source."""

    def _build(self, raw: _RawSample) -> DenseMatrixSample:
        if raw.array.ndim != 2:
            raise ValueError(
                f"Matrix sample from {raw.origin} must be a single 2D matrix, "
                f"got shape {raw.array.shape}"
            )
        return DenseMatrixSample(
            sample_id=raw.sample_id, matrix=np.asarray(raw.array, dtype=np.float64)
        )

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        """Load one matrix sample in dense float64 format."""
        return self._load(sample_id)

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        """Load one matrix sample as sparse COO components."""
        dense = self.load_dense_sample(sample_id)
        indices, values, size = _dense_to_sparse_components(dense.matrix)
        return SparseMatrixSample(
            sample_id=sample_id,
            indices=indices,
            values=values,
            size=size,
        )

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        """Iterate all dense matrix samples."""
        return self._iter()

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        """Iterate all sparse matrix samples."""
        for sample_id in self.sample_ids:
            yield self.load_sparse_sample(sample_id)


class _VectorStream(_SampleStream[VectorSample]):
    """`VectorSampleStream` implementation over any raw source."""

    def _build(self, raw: _RawSample) -> VectorSample:
        return VectorSample(
            sample_id=raw.sample_id, vector=_normalize_vector(raw.array, raw.origin)
        )

    def load_sample(self, sample_id: int) -> VectorSample:
        """Load one vector sample."""
        return self._load(sample_id)

    def iter_samples(self) -> Iterator[VectorSample]:
        """Iterate all vector samples."""
        return self._iter()


class NpyMatrixStream(_MatrixStream):
    """Matrix stream backed by a single .npy file with mmap."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            _NpyFileSource(
                path,
                noun="matrix",
                sample_ndim=2,
                shape_error="Matrix npy file must have shape (n,n) or (N,n,n), got {shape}",
            )
        )


class TxtMatrixStream(_MatrixStream):
    """Matrix stream backed by a single .txt file."""

    def __init__(self, path: Path) -> None:
        super().__init__(_TxtFileSource(path, noun="matrix"))


class GlobMatrixStream(_MatrixStream):
    """Matrix stream backed by glob-matched .txt/.npy files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
        include_indices: tuple[int, ...] | None = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            _GlobFileSource(
                expr,
                noun="matrix",
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
            )
        )


class NpyVectorStream(_VectorStream):
    """Vector stream backed by a .npy file with mmap."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            _NpyFileSource(
                path,
                noun="vector",
                sample_ndim=1,
                shape_error="Vector npy source must have shape (n,) or (N,n), got {shape}",
            )
        )


class TxtVectorStream(_VectorStream):
    """Vector stream backed by one .txt file."""

    def __init__(self, path: Path) -> None:
        super().__init__(_TxtFileSource(path, noun="vector"))


class GlobVectorStream(_VectorStream):
    """Vector stream backed by glob-matched .txt/.npy files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
        include_indices: tuple[int, ...] | None = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            _GlobFileSource(
                expr,
                noun="vector",
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
            )
        )


def open_matrix_stream(
    matrix_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    include_indices: tuple[int, ...] | None = None,
    exclude_indices: tuple[int, ...] = (),
) -> MatrixSampleStream:
    """Create a matrix sample stream from path expression."""
    if _is_glob_expression(matrix_path_expr):
        return GlobMatrixStream(
            matrix_path_expr,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
        )
    if include_indices is not None or exclude_indices:
        raise ValueError("include_indices/exclude_indices require a glob matrix source.")
    path = Path(matrix_path_expr)
    if not path.exists():
        raise FileNotFoundError(f"Matrix source not found: {path}")
    if path.suffix == ".npy":
        return NpyMatrixStream(path)
    if path.suffix == ".txt":
        return TxtMatrixStream(path)
    raise ValueError(
        f"Unsupported matrix source '{path}'. Supported: .txt, .npy, or glob patterns."
    )


def open_vector_stream(
    vector_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    include_indices: tuple[int, ...] | None = None,
    exclude_indices: tuple[int, ...] = (),
) -> VectorSampleStream:
    """Create a vector sample stream from path expression."""
    if _is_glob_expression(vector_path_expr):
        return GlobVectorStream(
            vector_path_expr,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
        )
    if include_indices is not None or exclude_indices:
        raise ValueError("include_indices/exclude_indices require a glob vector source.")
    path = Path(vector_path_expr)
    if not path.exists():
        raise FileNotFoundError(f"Vector source not found: {path}")
    if path.suffix == ".npy":
        return NpyVectorStream(path)
    if path.suffix == ".txt":
        return TxtVectorStream(path)
    raise ValueError(
        f"Unsupported vector source '{path}'. Supported: .txt, .npy, or glob patterns."
    )


def bind_sources(
    matrix_ids: tuple[int, ...],
    rhs_ids: tuple[int, ...] | None = None,
    solution_ids: tuple[int, ...] | None = None,
    parameters_ids_list: tuple[tuple[int, ...], ...] = (),
) -> list[SystemBinding]:
    """Bind matrix/rhs/solution/parameters ids with single-matrix broadcast semantics.

    Rules:
    - If only one matrix id exists and vectors have many ids, broadcast matrix id.
    - Otherwise bindings are keyed by matrix ids.
    - Provided rhs ids must match matrix ids (except single-matrix broadcast).
    - Provided solution ids must match matrix ids (except single-matrix broadcast).
    - Each entry in parameters_ids_list is a tuple of sample IDs for one parameter stream.

    Args:
        matrix_ids: Sample IDs from the matrix stream.
        rhs_ids: Optional sample IDs from the RHS stream.
        solution_ids: Optional sample IDs from the solution stream.
        parameters_ids_list: Tuple of ID tuples, one per parameter stream.

    Returns:
        List of ``SystemBinding`` objects with all sources resolved.
    """
    if not matrix_ids:
        raise ValueError("No matrix samples available to bind.")

    matrix_set = set(matrix_ids)
    rhs_set = set(rhs_ids or ())
    solution_set = set(solution_ids or ())
    param_sets = [set(ids) for ids in parameters_ids_list]

    if len(matrix_set) == 1:
        matrix_sample_id = next(iter(matrix_set))
        candidate_ids = (
            rhs_set if rhs_set else (solution_set if solution_set else {matrix_sample_id})
        )
        bound_ids = sorted(candidate_ids)
        return [
            SystemBinding(
                sample_id=sample_id,
                matrix_sample_id=matrix_sample_id,
                rhs_sample_id=sample_id if sample_id in rhs_set else None,
                solution_sample_id=sample_id if sample_id in solution_set else None,
                parameters_sample_ids=tuple(
                    sample_id if sample_id in ps else None for ps in param_sets
                ),
            )
            for sample_id in bound_ids
        ]

    bound_ids = sorted(matrix_set)
    if rhs_set and rhs_set != matrix_set:
        missing = sorted(matrix_set - rhs_set)
        extra = sorted(rhs_set - matrix_set)
        raise ValueError(
            f"RHS IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    if solution_set and solution_set != matrix_set:
        missing = sorted(matrix_set - solution_set)
        extra = sorted(solution_set - matrix_set)
        raise ValueError(
            f"solution IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    return [
        SystemBinding(
            sample_id=sample_id,
            matrix_sample_id=sample_id,
            rhs_sample_id=sample_id if sample_id in rhs_set else None,
            solution_sample_id=sample_id if sample_id in solution_set else None,
            parameters_sample_ids=tuple(
                sample_id if sample_id in ps else None for ps in param_sets
            ),
        )
        for sample_id in bound_ids
    ]


__all__ = [
    "DenseMatrixSample",
    "EnumerateBy",
    "GlobMatrixStream",
    "GlobVectorStream",
    "MatrixSampleStream",
    "SparseMatrixSample",
    "SystemBinding",
    "VectorSample",
    "VectorSampleStream",
    "_enumerate_files",
    "bind_sources",
    "open_matrix_stream",
    "open_vector_stream",
]
