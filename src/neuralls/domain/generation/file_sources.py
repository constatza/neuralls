"""Raw per-sample file readers: locate sample files and read their raw arrays.

A source owns *where* samples come from and *how many* there are — not what
format they're in. Three cardinality shapes, not four formats:
- ``_SingleFileSource``: one path, exactly one sample (.txt, .mtx, ...).
- ``_NpyFileSource``: one path, one sample or a leading-axis stack of them.
- ``_GlobFileSource``: many discovered paths, one sample per path.
The stream wrapped around a source (see ``source_streams.py``) owns *what* a
sample means (dense matrix vs 1D vector). Each class here satisfies the
``_RawSampleSource`` protocol (defined in ``source_streams.py``, next to its
consumer) structurally, with no explicit inheritance.

None of these classes read bytes off disk themselves: every one takes a
reader callable and calls it on the path(s) it owns. Format dispatch
(npy/txt/npz/mtx/mtx.gz) and the eager-vs-memory-mapped choice both live in
``platform/storage/matrix_readers.py``; domain only decides what a path
*means* (how many samples, which id), never how to parse its bytes.

Two reader shapes exist, not one, because a single union-returning reader
would be a genuine mismatch for half of the sources below:
- ``MatrixReader`` (``Path -> ndarray | csr_array``): for ``_SingleFileSource``
  and ``_GlobFileSource``, whose format is discovered per path, not known
  up front (a glob may match a mix of .txt/.npy/.mtx files).
- ``DenseReader`` (``Path -> ndarray``): for ``_NpyFileSource``, whose format
  is always ``.npy`` and therefore always dense — a type the caller can rely
  on directly, with nothing left to narrow at the call site.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from .sample_ids import EnumerateBy, _build_glob_index

# Mirror, but are independently defined from, platform/storage/matrix_readers.py's
# MatrixReader/read_dense_npy: domain cannot import that module directly (AGENTS.md's
# dependency rule — domain and platform are siblings; only composition may import both).
type MatrixReader = Callable[[Path], np.ndarray | csr_array]
type DenseReader = Callable[[Path], np.ndarray]


@dataclass(frozen=True, slots=True)
class MatrixReaders:
    """The reader(s) a generation run's raw sample sources are built with.

    Attributes:
        generic: Reads any registered format as its natural representation
            (dense or sparse) — for sources whose format is discovered per
            path rather than known up front.
        dense: Reads a ``.npy`` file specifically, always as a dense array —
            for sources whose format (and therefore dense-ness) is already
            fixed at construction.
    """

    generic: MatrixReader
    dense: DenseReader


@dataclass(frozen=True, slots=True)
class _RawSample:
    """One sample's array exactly as read from disk, plus the file it came from.

    Attributes:
        sample_id: Sample id this array belongs to.
        array: Raw array (possibly memory-mapped, possibly not yet float64).
        origin: File the array was read from, used for error messages.
    """

    sample_id: int
    array: np.ndarray | csr_array
    origin: Path


class _NpyFileSource:
    """Samples from one .npy file holding a single sample or a stack of them."""

    def __init__(
        self,
        path: Path,
        *,
        noun: str,
        sample_ndim: int,
        shape_error: str,
        reader: DenseReader,
    ) -> None:
        """Open *path* and determine how many samples it holds.

        Args:
            path: Source .npy file.
            noun: Source kind used in error messages ("matrix" or "vector").
            sample_ndim: Rank of a single sample; rank ``sample_ndim + 1`` is
                read as a stack of samples along the leading axis.
            shape_error: Message template, formatted with ``shape``, raised when
                the file's rank is neither of the two accepted ones.
            reader: Reads the dense array at a path.
        """
        self._path = path
        self._noun = noun
        self._sample_ndim = sample_ndim
        self._array = reader(path)
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


class _SingleFileSource:
    """The single sample held by one file, read in full by a reader call.

    Used for every format with no "maybe a stack" ambiguity (.txt, .mtx):
    once format dispatch moved into the injected reader, a single-sample
    file source no longer has any format-specific behavior left to vary —
    it is exactly one path, exactly one sample, one reader call.
    """

    def __init__(self, path: Path, *, noun: str, reader: MatrixReader) -> None:
        self._path = path
        self._noun = noun
        self._reader = reader

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return (0,)

    def read(self, sample_id: int) -> _RawSample:
        if sample_id != 0:
            raise KeyError(f"Unknown {self._noun} sample id {sample_id} for {self._path}")
        array = self._reader(self._path)
        return _RawSample(sample_id=0, array=array, origin=self._path)


class _GlobFileSource:
    """Samples from glob-matched files, one sample per file."""

    def __init__(
        self,
        expr: str,
        *,
        noun: str,
        sample_id_regex: str | None,
        enumerate_by: EnumerateBy | None,
        include_indices: tuple[int, ...] | None,
        exclude_indices: tuple[int, ...],
        reader: MatrixReader,
    ) -> None:
        self._noun = noun
        self._reader = reader
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
        array = self._reader(path)
        return _RawSample(sample_id=sample_id, array=array, origin=path)
