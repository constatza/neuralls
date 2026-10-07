"""Format-name to backend lookup for sparse batch storage.

``backend_for`` is the only place where a storage format name selects code.
Callers pass the format name recorded in the dataset manifest, so adding a
format means adding one entry here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from neuralls.platform.sparse_io.hdf5_store import (
    HDF5_GROUP_KEY,
    Hdf5MemberSink,
    Hdf5SparseReader,
    Hdf5SparseWriter,
)
from neuralls.platform.sparse_io.protocol import (
    CsrMemberSink,
    SparseLocation,
    SparseReader,
    SparseStreamWriter,
    SparseWriter,
)
from neuralls.platform.sparse_io.stream import CsrStreamWriter
from neuralls.platform.sparse_io.zarr_store import (
    ZarrMemberSink,
    ZarrSparseReader,
    ZarrSparseWriter,
)
from neuralls.shared.types import LayoutType

_BACKENDS: Final[Mapping[str, tuple[SparseReader, SparseWriter]]] = MappingProxyType(
    {
        "zarr": (ZarrSparseReader(), ZarrSparseWriter()),
        "hdf5": (Hdf5SparseReader(), Hdf5SparseWriter()),
    }
)


_MEMBER_SINKS: Final[Mapping[str, Callable[[SparseLocation], CsrMemberSink]]] = MappingProxyType(
    {
        "zarr": lambda location: ZarrMemberSink(location.path),
        "hdf5": lambda location: Hdf5MemberSink(location.path, _hdf5_key(location)),
    }
)


def _hdf5_key(location: SparseLocation) -> str:
    return location.key if location.key is not None else HDF5_GROUP_KEY


def open_stream_writer(
    format_name: str,
    location: SparseLocation,
    *,
    planned_samples: int,
    layout: LayoutType,
) -> SparseStreamWriter:
    """Open an incremental CSR writer for a storage format.

    The writer replaces whatever already sits at ``location``. Nothing is
    visible as a complete container until ``close`` has checked the sample count.

    Args:
        format_name: Storage format name, for example ``"zarr"``.
        location: Container to write (zarr group directory, or hdf5 file and group key).
        planned_samples: Exact number of samples the stream will receive. Must be at least 1.
        layout: Storage layout, chosen by the caller. The writer does not infer it.

    Returns:
        A writer whose ``write_batch`` appends samples and whose ``close`` returns the summary.

    Raises:
        ValueError: If no streaming backend is registered for ``format_name``, or the
            plan is invalid.
    """
    open_sink = _MEMBER_SINKS.get(format_name)
    if open_sink is None:
        raise ValueError(f"no streamed sparse backend registered for format {format_name!r}")
    return CsrStreamWriter(open_sink(location), planned_samples, layout)


def backend_for(format_name: str) -> tuple[SparseReader, SparseWriter]:
    """Return the (reader, writer) pair registered for a storage format.

    Args:
        format_name: Storage format name, for example ``"zarr"``.

    Returns:
        The reader and writer backends for that format.

    Raises:
        ValueError: If no backend is registered for ``format_name``.
    """
    backend = _BACKENDS.get(format_name)
    if backend is None:
        raise ValueError(f"no sparse backend registered for format {format_name!r}")
    return backend
