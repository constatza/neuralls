"""Streamed CSR writes must store exactly what the whole-group sparse writer stores for the same samples."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import CsrComponents, to_components
from neuralls.platform.sparse_io.hdf5_store import HDF5_GROUP_KEY
from neuralls.platform.sparse_io.protocol import SparseLocation
from neuralls.platform.sparse_io.registry import backend_for, open_stream_writer
from neuralls.platform.storage.csr_storage import (
    describe_csr_matrix_group,
    open_csr_matrix_stream,
)
from neuralls.shared.types import LayoutType

MEMBER_PATH = "matrix"
ONE_SAMPLE_PER_BATCH = 1
TWO_SAMPLES_PER_BATCH = 2
FORMATS = ("zarr", "hdf5")


def _batched(samples: Sequence[csr_array], size: int) -> list[list[csr_array]]:
    return [list(samples[start : start + size]) for start in range(0, len(samples), size)]


def _location(format_name: str, root: Path, label: str) -> SparseLocation:
    if format_name == "zarr":
        return SparseLocation(path=root / label)
    return SparseLocation(path=root / f"{label}.h5", key=HDF5_GROUP_KEY)


def _read_all(format_name: str, location: SparseLocation, count: int) -> list[CsrComponents]:
    reader, _ = backend_for(format_name)
    return [reader.read_sample(location, index) for index in range(count)]


def _assert_components_equal(actual: CsrComponents, expected: CsrComponents) -> None:
    assert actual.shape == expected.shape
    assert actual.indptr.tolist() == expected.indptr.tolist()
    assert actual.indices.tolist() == expected.indices.tolist()
    assert actual.data.tolist() == expected.data.tolist()


def _buffered_zarr(root: Path, samples: Sequence[csr_array]) -> Path:
    """Reference zarr group written in one call by the whole-group sparse writer."""
    group_dir = root / "buffered"
    _buffered_sparse("zarr", SparseLocation(path=group_dir), samples)
    return group_dir


def _buffered_sparse(
    format_name: str, location: SparseLocation, samples: Sequence[csr_array]
) -> None:
    _, writer = backend_for(format_name)
    writer.write(location, [to_components(sample) for sample in samples])


def _stream(
    format_name: str,
    location: SparseLocation,
    samples: Sequence[csr_array],
    batch_size: int,
    layout: LayoutType,
) -> None:
    stream = open_stream_writer(format_name, location, planned_samples=len(samples), layout=layout)
    for batch in _batched(samples, batch_size):
        stream.write_batch(batch)
    summary = stream.close()
    assert summary.sample_count == len(samples)
    assert summary.layout is layout


def _assert_streamed_equals_buffered(
    format_name: str,
    root: Path,
    samples: Sequence[csr_array],
    batch_size: int,
    layout: LayoutType,
) -> None:
    buffered = _location(format_name, root, "buffered")
    streamed = _location(format_name, root, "streamed")
    _buffered_sparse(format_name, buffered, samples)
    _stream(format_name, streamed, samples, batch_size, layout)

    expected = _read_all(format_name, buffered, len(samples))
    actual = _read_all(format_name, streamed, len(samples))
    for got, want in zip(actual, expected, strict=True):
        _assert_components_equal(got, want)


# --- Ragged layout ----------------------------------------------------------


def test_streamed_ragged_equals_buffered_zarr(
    tmp_path: Path, per_sample_matrices: list[csr_array]
) -> None:
    buffered_dir = _buffered_zarr(tmp_path, per_sample_matrices)
    streamed_dir = tmp_path / "streamed"
    stream = open_csr_matrix_stream(
        "zarr",
        streamed_dir,
        planned_samples=len(per_sample_matrices),
        layout=LayoutType.MANY_MATRICES,
    )
    for batch in _batched(per_sample_matrices, ONE_SAMPLE_PER_BATCH):
        stream.write_batch(batch)
    stream.close()

    assert describe_csr_matrix_group(streamed_dir, member_path=MEMBER_PATH) == (
        describe_csr_matrix_group(buffered_dir, member_path=MEMBER_PATH)
    )
    expected = _read_all("zarr", SparseLocation(path=buffered_dir), len(per_sample_matrices))
    actual = _read_all("zarr", SparseLocation(path=streamed_dir), len(per_sample_matrices))
    for got, want in zip(actual, expected, strict=True):
        _assert_components_equal(got, want)


def test_streamed_ragged_equals_buffered_hdf5(
    tmp_path: Path, per_sample_matrices: list[csr_array]
) -> None:
    _assert_streamed_equals_buffered(
        "hdf5", tmp_path, per_sample_matrices, TWO_SAMPLES_PER_BATCH, LayoutType.MANY_MATRICES
    )


@pytest.mark.parametrize("format_name", FORMATS)
def test_streamed_ragged_batches_of_one_equal_batches_of_two(
    format_name: str, tmp_path: Path, per_sample_matrices: list[csr_array]
) -> None:
    single = _location(format_name, tmp_path, "single")
    paired = _location(format_name, tmp_path, "paired")
    _stream(
        format_name, single, per_sample_matrices, ONE_SAMPLE_PER_BATCH, LayoutType.MANY_MATRICES
    )
    _stream(
        format_name, paired, per_sample_matrices, TWO_SAMPLES_PER_BATCH, LayoutType.MANY_MATRICES
    )

    for got, want in zip(
        _read_all(format_name, paired, len(per_sample_matrices)),
        _read_all(format_name, single, len(per_sample_matrices)),
        strict=True,
    ):
        _assert_components_equal(got, want)


# --- Shared-pattern layout --------------------------------------------------


def test_streamed_shared_pattern_equals_buffered_zarr(
    tmp_path: Path, shared_pattern_matrices: list[csr_array]
) -> None:
    buffered_dir = _buffered_zarr(tmp_path, shared_pattern_matrices)
    streamed_dir = tmp_path / "streamed"
    stream = open_csr_matrix_stream(
        "zarr",
        streamed_dir,
        planned_samples=len(shared_pattern_matrices),
        layout=LayoutType.SHARED_PATTERN,
    )
    for batch in _batched(shared_pattern_matrices, TWO_SAMPLES_PER_BATCH):
        stream.write_batch(batch)
    stream.close()

    assert describe_csr_matrix_group(streamed_dir, member_path=MEMBER_PATH) == (
        describe_csr_matrix_group(buffered_dir, member_path=MEMBER_PATH)
    )
    expected = _read_all("zarr", SparseLocation(path=buffered_dir), len(shared_pattern_matrices))
    actual = _read_all("zarr", SparseLocation(path=streamed_dir), len(shared_pattern_matrices))
    for got, want in zip(actual, expected, strict=True):
        _assert_components_equal(got, want)


def test_streamed_shared_pattern_equals_buffered_hdf5(
    tmp_path: Path, shared_pattern_matrices: list[csr_array]
) -> None:
    _assert_streamed_equals_buffered(
        "hdf5", tmp_path, shared_pattern_matrices, TWO_SAMPLES_PER_BATCH, LayoutType.SHARED_PATTERN
    )


@pytest.mark.parametrize("format_name", FORMATS)
def test_streamed_shared_pattern_rejects_a_different_pattern(
    format_name: str,
    tmp_path: Path,
    shared_pattern_matrices: list[csr_array],
    per_sample_matrices: list[csr_array],
) -> None:
    stream = open_stream_writer(
        format_name,
        _location(format_name, tmp_path, "mismatch"),
        planned_samples=len(shared_pattern_matrices) + 1,
        layout=LayoutType.SHARED_PATTERN,
    )
    stream.write_batch(shared_pattern_matrices[:1])
    with pytest.raises(ValueError, match="pattern"):
        stream.write_batch(per_sample_matrices[:1])


# --- Failure modes ----------------------------------------------------------


@pytest.mark.parametrize("format_name", FORMATS)
def test_streamed_short_source_raises(
    format_name: str, tmp_path: Path, per_sample_matrices: list[csr_array]
) -> None:
    stream = open_stream_writer(
        format_name,
        _location(format_name, tmp_path, "short"),
        planned_samples=4,
        layout=LayoutType.MANY_MATRICES,
    )
    stream.write_batch(per_sample_matrices)
    with pytest.raises(ValueError, match="planned 4 .*wrote 3"):
        stream.close()


@pytest.mark.parametrize("format_name", FORMATS)
def test_streamed_no_samples_raises(format_name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        open_stream_writer(
            format_name,
            _location(format_name, tmp_path, "empty"),
            planned_samples=0,
            layout=LayoutType.MANY_MATRICES,
        )
