"""The matrix loader keeps one matrix in memory, the one of the binding being generated.

Bindings are visited in order, so a matrix is needed only while its own binding runs.
Caching more than the current matrix would hold every matrix of the run at once.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import _cached_matrix_loader
from neuralls.domain.generation.specs import DatasetSpec
from neuralls.shared.types import MatrixFormat

_MATRIX_SIZE = 3


class _FakeMatrixStream:
    """Matrix stream stand-in that counts loads and returns a fresh array per load."""

    def __init__(self) -> None:
        self.load_count = 0

    def load_sample(self, sample_id: int, matrix_format: MatrixFormat) -> np.ndarray:
        self.load_count += 1
        return np.eye(_MATRIX_SIZE, dtype=np.float64) * (sample_id + 1)


@pytest.fixture
def stream() -> _FakeMatrixStream:
    return _FakeMatrixStream()


def test_repeat_of_current_matrix_is_not_reloaded(stream: _FakeMatrixStream) -> None:
    get_matrix = _cached_matrix_loader(stream, DatasetSpec(), MatrixFormat.DENSE)

    first = get_matrix(0)
    second = get_matrix(0)

    assert second is first
    assert stream.load_count == 1


def test_matrix_of_a_finished_binding_is_released(stream: _FakeMatrixStream) -> None:
    get_matrix = _cached_matrix_loader(stream, DatasetSpec(), MatrixFormat.DENSE)

    finished = weakref.ref(get_matrix(0).matrix_norm)
    get_matrix(1)
    gc.collect()

    assert finished() is None


def test_returning_to_an_earlier_matrix_reloads_it(stream: _FakeMatrixStream) -> None:
    get_matrix = _cached_matrix_loader(stream, DatasetSpec(), MatrixFormat.DENSE)

    get_matrix(0)
    get_matrix(1)
    get_matrix(0)

    assert stream.load_count == 3
