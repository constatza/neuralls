"""Streaming source readers for matrix and vector sample ingestion.

This module keeps source discovery and per-sample loading decoupled from
generation strategy logic. It supports:
- single .txt/.npy/.mtx matrix files (vectors: .txt/.npy)
- single .npy matrix/vector files (including stacked samples via mmap)
- glob patterns of .txt/.npy/.mtx/.mtx.gz/.npz files, one sample per file,
  including directory-per-sample layouts (e.g. "raw/*/K_ff.mtx")

Every stream/source here takes an injected ``readers: MatrixReaders`` (see
``file_sources.py``) and never parses bytes itself — format dispatch lives
entirely in ``platform/storage/matrix_readers.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import csr_array

from neuralls.shared.types import MatrixFormat, SystemMatrix

from .file_sources import (
    DenseReader,
    MatrixReader,
    MatrixReaders,
    _GlobFileSource,
    _NpyFileSource,
    _RawSample,
    _SingleFileSource,
)
from .sample_ids import EnumerateBy, _is_glob_expression


def _normalize_vector(array: np.ndarray, source: Path) -> np.ndarray:
    """Normalize vector shape to 1D float64."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.reshape(-1)
    raise ValueError(f"Expected vector from {source}, got shape {arr.shape}")


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

    def load_sample(self, sample_id: int, matrix_format: MatrixFormat) -> SystemMatrix:
        """Load one matrix sample in the requested storage format."""
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


class _RawSampleSource(Protocol):
    """Protocol for locating sample files and reading their raw arrays.

    A source owns *where* samples come from (one stacked .npy, one .txt, a
    glob of per-sample files, or a MatrixMarket .mtx file); the stream wrapped
    around it owns *what* a sample means (dense matrix vs 1D vector). The
    concrete sources in ``file_sources.py`` satisfy this structurally, with
    no explicit inheritance.
    """

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def read(self, sample_id: int) -> _RawSample:
        """Read one sample's raw array."""
        ...


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


def _check_matrix_raw(raw: _RawSample) -> None:
    """Reject raw arrays that are not a single 2D matrix."""
    if raw.array.ndim != 2:
        raise ValueError(
            f"Matrix sample from {raw.origin} must be a single 2D matrix, "
            f"got shape {raw.array.shape}"
        )


class _MatrixStream(_SampleStream[DenseMatrixSample]):
    """`MatrixSampleStream` implementation over any raw source."""

    def _build(self, raw: _RawSample) -> DenseMatrixSample:
        _check_matrix_raw(raw)
        # Densifying a CSR raw array is the explicit dense-request path only.
        dense = raw.array.toarray() if isinstance(raw.array, csr_array) else raw.array
        return DenseMatrixSample(
            sample_id=raw.sample_id, matrix=np.asarray(dense, dtype=np.float64)
        )

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        """Load one matrix sample in dense float64 format."""
        return self._load(sample_id)

    def load_sample(self, sample_id: int, matrix_format: MatrixFormat) -> SystemMatrix:
        """Load one matrix sample in the requested storage format.

        CSR input is never densified: a CSR raw array is returned as CSR, and a
        dense raw array requested as CSR is converted once with ``csr_array``.
        """
        match matrix_format:
            case MatrixFormat.DENSE:
                return self.load_dense_sample(sample_id).matrix
            case MatrixFormat.CSR:
                raw = self._source.read(sample_id)
                _check_matrix_raw(raw)
                if isinstance(raw.array, csr_array):
                    return csr_array(raw.array, dtype=np.float64)
                return csr_array(np.asarray(raw.array, dtype=np.float64))

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
        if isinstance(raw.array, csr_array):
            raise TypeError(f"Vector sample from {raw.origin} cannot be a sparse matrix")
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

    def __init__(self, path: Path, *, reader: DenseReader) -> None:
        super().__init__(
            _NpyFileSource(
                path,
                noun="matrix",
                sample_ndim=2,
                shape_error="Matrix npy file must have shape (n,n) or (N,n,n), got {shape}",
                reader=reader,
            )
        )


class TxtMatrixStream(_MatrixStream):
    """Matrix stream backed by a single .txt file."""

    def __init__(self, path: Path, *, reader: MatrixReader) -> None:
        super().__init__(_SingleFileSource(path, noun="matrix", reader=reader))


class GlobMatrixStream(_MatrixStream):
    """Matrix stream backed by glob-matched files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
        include_indices: tuple[int, ...] | None = None,
        exclude_indices: tuple[int, ...] = (),
        *,
        reader: MatrixReader,
    ) -> None:
        super().__init__(
            _GlobFileSource(
                expr,
                noun="matrix",
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
                reader=reader,
            )
        )


class MtxMatrixStream(_MatrixStream):
    """Matrix stream backed by a single MatrixMarket (.mtx) file."""

    def __init__(self, path: Path, *, reader: MatrixReader) -> None:
        super().__init__(_SingleFileSource(path, noun="matrix", reader=reader))


class NpyVectorStream(_VectorStream):
    """Vector stream backed by a .npy file with mmap."""

    def __init__(self, path: Path, *, reader: DenseReader) -> None:
        super().__init__(
            _NpyFileSource(
                path,
                noun="vector",
                sample_ndim=1,
                shape_error="Vector npy source must have shape (n,) or (N,n), got {shape}",
                reader=reader,
            )
        )


class TxtVectorStream(_VectorStream):
    """Vector stream backed by one .txt file."""

    def __init__(self, path: Path, *, reader: MatrixReader) -> None:
        super().__init__(_SingleFileSource(path, noun="vector", reader=reader))


class GlobVectorStream(_VectorStream):
    """Vector stream backed by glob-matched files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
        include_indices: tuple[int, ...] | None = None,
        exclude_indices: tuple[int, ...] = (),
        *,
        reader: MatrixReader,
    ) -> None:
        super().__init__(
            _GlobFileSource(
                expr,
                noun="vector",
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
                reader=reader,
            )
        )


@dataclass(frozen=True, slots=True)
class _GlobExpr:
    """A path expression whose wildcard may match any number of files."""

    expr: str


@dataclass(frozen=True, slots=True)
class _SingleFileExpr:
    """A path expression naming exactly one, already-existing file."""

    path: Path


_PathExpr = _GlobExpr | _SingleFileExpr


def _resolve_path_expr(expr: str, *, noun: str) -> _PathExpr:
    """Classify a config path expression by cardinality: glob or single file.

    A glob may match any number of files; a plain path names exactly one,
    which must already exist. This is the one place that decides which of
    the two an expression is — callers match on the result instead of
    re-deriving the distinction with their own `_is_glob_expression` check.
    """
    if _is_glob_expression(expr):
        return _GlobExpr(expr)
    path = Path(expr)
    if not path.exists():
        raise FileNotFoundError(f"{noun.capitalize()} source not found: {path}")
    return _SingleFileExpr(path)


def open_matrix_stream(
    matrix_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    include_indices: tuple[int, ...] | None = None,
    exclude_indices: tuple[int, ...] = (),
    *,
    readers: MatrixReaders,
) -> MatrixSampleStream:
    """Create a matrix sample stream from path expression."""
    match _resolve_path_expr(matrix_path_expr, noun="matrix"):
        case _GlobExpr(expr):
            return GlobMatrixStream(
                expr,
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
                reader=readers.generic,
            )
        case _SingleFileExpr(path):
            if include_indices is not None or exclude_indices:
                raise ValueError("include_indices/exclude_indices require a glob matrix source.")
            match path.suffix:
                case ".npy":
                    return NpyMatrixStream(path, reader=readers.dense)
                case ".txt":
                    return TxtMatrixStream(path, reader=readers.generic)
                case ".mtx":
                    return MtxMatrixStream(path, reader=readers.generic)
                case _:
                    raise ValueError(
                        f"Unsupported matrix source '{path}'. "
                        "Supported: .txt, .npy, .mtx, or glob patterns."
                    )


def open_vector_stream(
    vector_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    include_indices: tuple[int, ...] | None = None,
    exclude_indices: tuple[int, ...] = (),
    *,
    readers: MatrixReaders,
) -> VectorSampleStream:
    """Create a vector sample stream from path expression."""
    match _resolve_path_expr(vector_path_expr, noun="vector"):
        case _GlobExpr(expr):
            return GlobVectorStream(
                expr,
                sample_id_regex=sample_id_regex,
                enumerate_by=enumerate_by,
                include_indices=include_indices,
                exclude_indices=exclude_indices,
                reader=readers.generic,
            )
        case _SingleFileExpr(path):
            if include_indices is not None or exclude_indices:
                raise ValueError("include_indices/exclude_indices require a glob vector source.")
            match path.suffix:
                case ".npy":
                    return NpyVectorStream(path, reader=readers.dense)
                case ".txt":
                    return TxtVectorStream(path, reader=readers.generic)
                case _:
                    raise ValueError(
                        f"Unsupported vector source '{path}'. "
                        "Supported: .txt, .npy, or glob patterns."
                    )


__all__ = [
    "DenseMatrixSample",
    "EnumerateBy",
    "GlobMatrixStream",
    "GlobVectorStream",
    "MatrixSampleStream",
    "SparseMatrixSample",
    "VectorSample",
    "VectorSampleStream",
    "open_matrix_stream",
    "open_vector_stream",
]
