"""Tests for the storage-agnostic sparse reader/writer contracts and the registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import CsrComponents, to_components
from neuralls.platform.sparse_io.hdf5_store import Hdf5SparseReader, Hdf5SparseWriter
from neuralls.platform.sparse_io.protocol import (
    SparseLocation,
    SparseReader,
    SparseSummary,
    SparseWriter,
)
from neuralls.platform.sparse_io.registry import backend_for
from neuralls.platform.sparse_io.zarr_store import ZarrSparseReader, ZarrSparseWriter
from neuralls.shared.types import LayoutType

IN_MEMORY_KEY = "matrix"


@dataclass
class InMemorySparseBackend:
    """Fake backend keeping samples in a list; satisfies both protocols structurally."""

    samples: list[CsrComponents] = field(default_factory=list)

    def summary(self, location: SparseLocation) -> SparseSummary:
        return SparseSummary(sample_count=len(self.samples), layout=LayoutType.MANY_MATRICES)

    def read_sample(self, location: SparseLocation, sample_index: int) -> CsrComponents:
        return self.samples[sample_index]

    def write(self, location: SparseLocation, samples: Sequence[CsrComponents]) -> SparseSummary:
        self.samples = list(samples)
        return self.summary(location)


@pytest.fixture
def location(tmp_path: Path) -> SparseLocation:
    return SparseLocation(path=tmp_path / "matrices.store", key=IN_MEMORY_KEY)


@pytest.fixture
def stored_backend(csr_with_empty_rows_and_cols: csr_array) -> InMemorySparseBackend:
    return InMemorySparseBackend(samples=[to_components(csr_with_empty_rows_and_cols)])


def test_fake_backend_satisfies_reader_protocol(
    stored_backend: InMemorySparseBackend, location: SparseLocation
) -> None:
    reader: SparseReader = stored_backend
    summary = reader.summary(location)
    sample = reader.read_sample(location, 0)

    assert summary == SparseSummary(sample_count=1, layout=LayoutType.MANY_MATRICES)
    assert sample.shape == (4, 6)


def test_fake_backend_satisfies_writer_protocol(
    stored_backend: InMemorySparseBackend,
    location: SparseLocation,
    csr_with_stored_zero: csr_array,
) -> None:
    writer: SparseWriter = stored_backend
    summary = writer.write(location, [to_components(csr_with_stored_zero)])

    assert summary.sample_count == 1
    assert stored_backend.samples[0].shape == (2, 2)


def test_backend_for_unregistered_format_names_the_format() -> None:
    with pytest.raises(ValueError, match="npy"):
        backend_for("npy")


def test_backend_for_zarr_returns_zarr_reader_and_writer() -> None:
    reader, writer = backend_for("zarr")
    assert isinstance(reader, ZarrSparseReader)
    assert isinstance(writer, ZarrSparseWriter)


def test_backend_for_hdf5_returns_hdf5_reader_and_writer() -> None:
    reader, writer = backend_for("hdf5")
    assert isinstance(reader, Hdf5SparseReader)
    assert isinstance(writer, Hdf5SparseWriter)
