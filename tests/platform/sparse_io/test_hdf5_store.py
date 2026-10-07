"""Tests for the hdf5 sparse writer and reader."""

from __future__ import annotations

from collections.abc import Sequence

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import CsrComponents, to_components
from neuralls.platform.sparse_io.hdf5_store import (
    HDF5_GROUP_KEY,
    Hdf5SparseReader,
    Hdf5SparseWriter,
)
from neuralls.platform.sparse_io.layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
    pack_per_sample,
    pack_shared_pattern,
)
from neuralls.platform.sparse_io.protocol import SparseLocation
from neuralls.shared.types import LayoutType


def _assert_components_equal(actual: CsrComponents, expected: CsrComponents) -> None:
    assert actual.shape == expected.shape
    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.indices, expected.indices)
    assert np.array_equal(actual.data, expected.data)


def _components(matrices: Sequence[csr_array]) -> list[CsrComponents]:
    return [to_components(matrix) for matrix in matrices]


def _assert_member(group: h5py.Group, name: str, expected: np.ndarray) -> None:
    dataset = group[name]
    assert isinstance(dataset, h5py.Dataset)
    assert dataset.dtype == expected.dtype
    assert dataset.shape == expected.shape
    assert np.array_equal(dataset[()], expected)


# --- Step 3: writer -------------------------------------------------------


def test_writer_per_sample_members_match_pack(
    hdf5_location: SparseLocation, per_sample_matrices: list[csr_array]
) -> None:
    Hdf5SparseWriter().write(hdf5_location, _components(per_sample_matrices))
    expected = pack_per_sample(per_sample_matrices)

    with h5py.File(hdf5_location.path, "r") as file:
        group = file[HDF5_GROUP_KEY]
        assert isinstance(group, h5py.Group)
        _assert_member(group, INDPTR_ARRAY, expected.indptr)
        _assert_member(group, INDICES_ARRAY, expected.indices)
        _assert_member(group, DATA_ARRAY, expected.data)
        _assert_member(group, SAMPLE_OFFSETS_ARRAY, expected.sample_offsets)
        _assert_member(group, SHAPE_ARRAY, expected.shape)


def test_writer_shared_pattern_members_match_pack(
    hdf5_location: SparseLocation, shared_pattern_matrices: list[csr_array]
) -> None:
    Hdf5SparseWriter().write(hdf5_location, _components(shared_pattern_matrices))
    expected = pack_shared_pattern(shared_pattern_matrices)

    with h5py.File(hdf5_location.path, "r") as file:
        group = file[HDF5_GROUP_KEY]
        assert isinstance(group, h5py.Group)
        _assert_member(group, INDPTR_ARRAY, expected.indptr)
        _assert_member(group, INDICES_ARRAY, expected.indices)
        _assert_member(group, DATA_ARRAY, expected.data)
        _assert_member(group, SHAPE_ARRAY, expected.shape)
        assert SAMPLE_OFFSETS_ARRAY not in group


def test_writer_returns_summary_with_layout(
    hdf5_location: SparseLocation,
    per_sample_matrices: list[csr_array],
    shared_pattern_matrices: list[csr_array],
) -> None:
    writer = Hdf5SparseWriter()

    per_sample = writer.write(hdf5_location, _components(per_sample_matrices))
    shared = writer.write(hdf5_location, _components(shared_pattern_matrices))

    assert (per_sample.sample_count, per_sample.layout) == (3, LayoutType.MANY_MATRICES)
    assert (shared.sample_count, shared.layout) == (4, LayoutType.SHARED_PATTERN)


def test_writer_overwrites_existing_group(
    hdf5_location: SparseLocation,
    per_sample_matrices: list[csr_array],
    shared_pattern_matrices: list[csr_array],
) -> None:
    writer = Hdf5SparseWriter()
    writer.write(hdf5_location, _components(per_sample_matrices))
    writer.write(hdf5_location, _components(shared_pattern_matrices))

    with h5py.File(hdf5_location.path, "r") as file:
        group = file[HDF5_GROUP_KEY]
        assert isinstance(group, h5py.Group)
        assert SAMPLE_OFFSETS_ARRAY not in group
        assert group[SHAPE_ARRAY].shape == (4, 2)


def test_writer_rejects_empty_sample_list(hdf5_location: SparseLocation) -> None:
    with pytest.raises(ValueError, match="empty"):
        Hdf5SparseWriter().write(hdf5_location, [])


# --- Step 4: reader -------------------------------------------------------


def test_reader_round_trips_per_sample_layout(
    hdf5_location: SparseLocation, per_sample_matrices: list[csr_array]
) -> None:
    samples = _components(per_sample_matrices)
    Hdf5SparseWriter().write(hdf5_location, samples)
    reader = Hdf5SparseReader()

    for index, expected in enumerate(samples):
        _assert_components_equal(reader.read_sample(hdf5_location, index), expected)


def test_reader_round_trips_shared_pattern_layout(
    hdf5_location: SparseLocation, shared_pattern_matrices: list[csr_array]
) -> None:
    samples = _components(shared_pattern_matrices)
    Hdf5SparseWriter().write(hdf5_location, samples)
    reader = Hdf5SparseReader()

    for index, expected in enumerate(samples):
        _assert_components_equal(reader.read_sample(hdf5_location, index), expected)


def test_reader_summary_per_sample(
    hdf5_location: SparseLocation, per_sample_matrices: list[csr_array]
) -> None:
    Hdf5SparseWriter().write(hdf5_location, _components(per_sample_matrices))

    summary = Hdf5SparseReader().summary(hdf5_location)

    assert summary.sample_count == 3
    assert summary.layout is LayoutType.MANY_MATRICES


def test_reader_summary_shared_pattern(
    hdf5_location: SparseLocation, shared_pattern_matrices: list[csr_array]
) -> None:
    Hdf5SparseWriter().write(hdf5_location, _components(shared_pattern_matrices))

    summary = Hdf5SparseReader().summary(hdf5_location)

    assert summary.sample_count == 4
    assert summary.layout is LayoutType.SHARED_PATTERN


@pytest.mark.parametrize("bad_index", [-1, 3])
def test_reader_out_of_range_index_raises(
    hdf5_location: SparseLocation, per_sample_matrices: list[csr_array], bad_index: int
) -> None:
    Hdf5SparseWriter().write(hdf5_location, _components(per_sample_matrices))

    with pytest.raises(IndexError):
        Hdf5SparseReader().read_sample(hdf5_location, bad_index)
