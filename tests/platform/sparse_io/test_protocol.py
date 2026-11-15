"""The sparse backend registry returns the reader and writer of each registered format."""

from __future__ import annotations

import pytest

from neuralls.platform.sparse_io.hdf5_store import Hdf5SparseReader, Hdf5SparseWriter
from neuralls.platform.sparse_io.registry import backend_for
from neuralls.platform.sparse_io.zarr_store import ZarrSparseReader, ZarrSparseWriter


def test_backend_for_unregistered_format_raises() -> None:
    with pytest.raises(ValueError):
        backend_for("npy")


@pytest.mark.parametrize(
    ("format_name", "reader_type", "writer_type"),
    [
        ("zarr", ZarrSparseReader, ZarrSparseWriter),
        ("hdf5", Hdf5SparseReader, Hdf5SparseWriter),
    ],
)
def test_backend_for_returns_the_reader_and_writer_of_the_format(
    format_name: str, reader_type: type, writer_type: type
) -> None:
    reader, writer = backend_for(format_name)

    assert (type(reader), type(writer)) == (reader_type, writer_type)
