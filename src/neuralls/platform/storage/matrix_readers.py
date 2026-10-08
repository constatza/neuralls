"""Suffix-keyed loaders for system matrices.

Every on-disk matrix format maps to one loader in ``MATRIX_READERS``. Callers
go through ``read_matrix`` and never branch on the file suffix themselves, so
adding a format means adding one loader function and one mapping entry.

Loaders return either a dense ``np.ndarray`` (dense text and binary formats)
or a ``csr_array`` (sparse formats). ``to_dense`` converts explicitly for
callers that still require a dense array.

I/O actions - these functions read from disk.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_array, load_npz

from neuralls.shared.types import SystemMatrix

NPY_SUFFIX = ".npy"
TXT_SUFFIX = ".txt"
NPZ_SUFFIX = ".npz"
MTX_SUFFIX = ".mtx"
MTX_GZ_SUFFIX = ".mtx.gz"

type MatrixReader = Callable[[Path], SystemMatrix]


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64, copy=False)


def _read_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _read_npz(path: Path) -> csr_array:
    try:
        loaded = load_npz(path)
    except ValueError as exc:
        raise ValueError(
            f"{path} does not contain a sparse matrix; .npz matrices must be scipy sparse archives"
        ) from exc
    return csr_array(loaded)


def _read_mtx(path: Path) -> csr_array:
    return csr_array(mmread(path, spmatrix=False))


def _read_mtx_gz(path: Path) -> csr_array:
    with gzip.open(path, "rb") as stream:
        payload = io.BytesIO(stream.read())
    return csr_array(mmread(payload, spmatrix=False))


MATRIX_READERS: Mapping[str, MatrixReader] = MappingProxyType(
    {
        NPY_SUFFIX: _read_npy,
        TXT_SUFFIX: _read_txt,
        NPZ_SUFFIX: _read_npz,
        MTX_SUFFIX: _read_mtx,
        MTX_GZ_SUFFIX: _read_mtx_gz,
    }
)


def _read_npy_lazy(path: Path) -> np.ndarray:
    # No dtype cast here: casting would copy the array, defeating the mmap.
    return np.load(path, mmap_mode="r")


_LAZY_MATRIX_READERS: Mapping[str, MatrixReader] = MappingProxyType(
    {
        NPY_SUFFIX: _read_npy_lazy,
    }
)


def _suffix_key(path: Path) -> str:
    """Return the registry key for a path; compound suffixes take precedence."""
    if path.name.endswith(MTX_GZ_SUFFIX):
        return MTX_GZ_SUFFIX
    return path.suffix


def read_matrix(path: Path, *, lazy: bool = False) -> SystemMatrix:
    """Load a system matrix, dispatching on its file suffix.

    I/O action - reads the matrix file from disk.

    Args:
        path: Path to a matrix file with a registered suffix
        lazy: Memory-map `.npy` instead of eagerly loading and casting it.
            Every other format has no partial-read mechanism in the library
            that parses it, so this has no effect on them.

    Returns:
        Dense ndarray or CSR array, depending on the format's loader

    Raises:
        ValueError: If the suffix is not registered in ``MATRIX_READERS``
    """
    key = _suffix_key(path)
    reader = (lazy and _LAZY_MATRIX_READERS.get(key)) or MATRIX_READERS.get(key)
    if reader is None:
        supported = ", ".join(sorted(MATRIX_READERS))
        raise ValueError(
            f"Unsupported matrix file suffix for {path}; supported suffixes: {supported}"
        )
    return reader(path)


def read_dense_npy(path: Path, *, lazy: bool = False) -> np.ndarray:
    """Read a ``.npy`` file, always returning a dense array.

    Narrower than ``read_matrix``: unlike the general suffix-dispatched
    reader, this one's return type carries no sparse branch, because `.npy`
    never decodes to one. Callers that already know they're reading `.npy`
    (and only `.npy`) should use this instead of narrowing `read_matrix`'s
    result themselves.
    """
    return (_read_npy_lazy if lazy else _read_npy)(path)


def to_dense(matrix: SystemMatrix) -> np.ndarray:
    """Convert a system matrix to a dense ndarray.

    Pure function. Dense input is returned unchanged.

    Args:
        matrix: Dense ndarray or CSR array

    Returns:
        Dense ndarray with the same values
    """
    if isinstance(matrix, np.ndarray):
        return matrix
    return matrix.toarray()
