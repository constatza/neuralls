"""SampleWriter writes batches at a running offset through an injected ArrayStore."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.sample_writer import DatasetArtifacts, SampleWriter

_N = 3
_P = 2
_MATRIX_ID = 7


class InMemoryArrayStore:
    """ArrayStore fake that keeps every named array in a dict and tracks written rows."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.written: dict[str, int] = {}
        self.closed = False

    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        if name in self.arrays:
            raise ValueError(f"array {name!r} already exists")
        self.arrays[name] = np.zeros(shape, dtype=dtype)
        self.written[name] = 0

    def write_rows(self, name: str, start: int, data: np.ndarray) -> None:
        rows = data.shape[0]
        self.arrays[name][start : start + rows] = data
        self.written[name] += rows

    def close(self) -> None:
        self.closed = True


def _batch(
    binding_index: int,
    rows: int,
    offset: float,
    *,
    matrix_sample_id: int = _MATRIX_ID,
    vectors: tuple[np.ndarray | None, ...] = (),
) -> SampleBatch:
    """Batch whose rhs values encode the running offset so placement is checkable."""
    base = np.arange(rows * _N, dtype=np.float64).reshape(rows, _N) + offset
    return SampleBatch(
        binding_index=binding_index,
        strategy_name="gaussian_forward",
        rhs=base,
        solutions=-base,
        row_kind_codes=np.full(rows, 2, dtype=np.uint8),
        matrix_sample_index=np.full(rows, matrix_sample_id, dtype=np.int64),
        parameter_vectors=vectors,
    )


@pytest.fixture
def matrix_for() -> object:
    """Dense 3x3 matrix per sample id, distinct for each id."""
    return lambda sample_id: np.eye(_N, dtype=np.float64) * (sample_id + 1)


@pytest.fixture
def first_batch() -> SampleBatch:
    return _batch(0, 2, 0.0, vectors=(np.array([0.5, 1.5]), None))


@pytest.fixture
def second_batch() -> SampleBatch:
    return _batch(0, 3, 100.0, vectors=(np.array([0.5, 1.5]), None))


def test_write_batch_then_finalize_places_rows_at_running_offset(
    first_batch: SampleBatch, second_batch: SampleBatch, matrix_for: object
) -> None:
    store = InMemoryArrayStore()
    writer = SampleWriter(store, planned_rows=5, matrix_for=matrix_for, single_matrix=False)

    writer.write_batch(first_batch)
    writer.write_batch(second_batch)
    artifacts = writer.finalize()

    np.testing.assert_array_equal(store.arrays["rhs"][:2], first_batch.rhs)
    np.testing.assert_array_equal(store.arrays["rhs"][2:], second_batch.rhs)
    np.testing.assert_array_equal(store.arrays["solutions"][2:], second_batch.solutions)
    np.testing.assert_array_equal(store.arrays["row_kind"], np.full(5, 2, dtype=np.uint8))
    np.testing.assert_array_equal(store.arrays["matrix_sample_index"], np.full(5, _MATRIX_ID))
    np.testing.assert_array_equal(
        store.arrays["parameters_0"], np.tile(np.array([0.5, 1.5]), (5, 1))
    )
    assert "parameters_1" not in store.arrays
    np.testing.assert_array_equal(
        store.arrays["matrix"], np.repeat(np.eye(_N)[np.newaxis] * (_MATRIX_ID + 1), 5, axis=0)
    )
    assert store.closed
    assert isinstance(artifacts, DatasetArtifacts)
    assert {entry.name: entry.rows for entry in artifacts.arrays} == {
        "rhs": 5,
        "solutions": 5,
        "row_kind": 5,
        "matrix_sample_index": 5,
        "matrix": 5,
        "parameters_0": 5,
    }


def test_finalize_raises_when_written_rows_differ_from_plan(
    first_batch: SampleBatch, matrix_for: object
) -> None:
    store = InMemoryArrayStore()
    writer = SampleWriter(store, planned_rows=5, matrix_for=matrix_for, single_matrix=False)
    writer.write_batch(first_batch)

    with pytest.raises(ValueError, match="planned 5"):
        writer.finalize()


def test_single_matrix_layout_writes_one_matrix_row(
    first_batch: SampleBatch, matrix_for: object
) -> None:
    store = InMemoryArrayStore()
    writer = SampleWriter(store, planned_rows=2, matrix_for=matrix_for, single_matrix=True)
    writer.write_batch(first_batch)

    artifacts = writer.finalize()

    assert store.arrays["matrix"].shape == (1, _N, _N)
    assert {entry.name: entry.rows for entry in artifacts.arrays}["matrix"] == 1


def test_empty_batch_writes_nothing(matrix_for: object) -> None:
    store = InMemoryArrayStore()
    writer = SampleWriter(store, planned_rows=0, matrix_for=matrix_for, single_matrix=False)
    writer.write_batch(_batch(0, 0, 0.0))

    with pytest.raises(ValueError, match="No samples"):
        writer.finalize()
    assert store.written == {}


def test_writer_exposes_no_read_methods() -> None:
    read_like = [
        name
        for name in dir(SampleWriter)
        if not name.startswith("_") and name.startswith(("read_", "get_"))
    ]
    assert read_like == []
    assert not hasattr(SampleWriter, "read")
    assert not hasattr(SampleWriter, "get")
