"""Raw per-sample file readers: locate sample files and read their raw arrays.

A source owns *where* samples come from (one stacked .npy, one .txt, a glob of
per-sample files, or a MatrixMarket .mtx file); the stream wrapped around it
(see ``source_streams.py``) owns *what* a sample means (dense matrix vs 1D vector).
Each class here satisfies the ``_RawSampleSource`` protocol (defined in
``source_streams.py``, next to its consumer) structurally, with no explicit
inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_array

from .sample_ids import EnumerateBy, _build_glob_index


@dataclass(frozen=True)
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


class _MtxFileSource:
    """The single sample held by one MatrixMarket (.mtx) file, read as CSR."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return (0,)

    def read(self, sample_id: int) -> _RawSample:
        if sample_id != 0:
            raise KeyError(f"Unknown matrix sample id {sample_id} for {self._path}")
        array = csr_array(mmread(self._path, spmatrix=False))
        return _RawSample(sample_id=0, array=array, origin=self._path)
