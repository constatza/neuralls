"""Fixtures for the dataset commit/staging and CSR build tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from tests import atomic_support

from neuralls.composition.generation import dense_streaming
from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.sample_writer import SampleWriter
from neuralls.shared.types import DatasetFormat

# Re-exported so these tests share the seeded dense-generation fixtures with the other conftest.
atomic_matrix_dir = atomic_support.atomic_matrix_dir
atomic_source = atomic_support.atomic_source
atomic_spec = atomic_support.atomic_spec


@pytest.fixture(params=["zarr", "hdf5"])
def dense_format(request: pytest.FixtureRequest) -> DatasetFormat:
    return request.param


@pytest.fixture
def interrupt_writer(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Make the streamed writer raise after ``n`` batches have been written."""

    def _install(after_batches: int) -> None:
        written = 0

        class _InterruptedWriter(SampleWriter):
            def write_batch(self, batch: SampleBatch) -> None:
                nonlocal written
                if written >= after_batches:
                    raise RuntimeError("injected interruption")
                written += 1
                super().write_batch(batch)

        monkeypatch.setattr(dense_streaming, "SampleWriter", _InterruptedWriter)

    return _install
