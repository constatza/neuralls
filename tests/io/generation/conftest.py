"""Fixtures for the dataset commit/staging and CSR build tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tests import atomic_support, generation_support
from tests.dataset_support import build_test_dataset

from neuralls.composition.generation import dense_streaming
from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.sample_writer import SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.shared.types import DatasetFormat

# Re-exported so these tests share the seeded dense-generation fixtures with the other conftest.
atomic_matrix_dir = atomic_support.atomic_matrix_dir
atomic_source = atomic_support.atomic_source
atomic_spec = atomic_support.atomic_spec

# Re-exported so these tests share the fixtures with tests/generation.
solver_overrides = generation_support.solver_overrides
residual_solver = generation_support.residual_solver
warning_messages = generation_support.warning_messages
three_spd_matrix_dir = generation_support.three_spd_matrix_dir
solution_block = generation_support.solution_block
open_binding_stream = generation_support.open_binding_stream
subdomain_matrices = generation_support.subdomain_matrices


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


@pytest.fixture
def spd_matrix_file(tmp_path: Path) -> Path:
    """A small symmetric positive-definite matrix on disk, as a generation source."""
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    sources = tmp_path / "sources"
    sources.mkdir()
    matrix_file = sources / "matrix.txt"
    np.savetxt(matrix_file, matrix)
    return matrix_file


@pytest.fixture
def source_spec(spd_matrix_file: Path) -> SourceSpec:
    """Source reading the single on-disk matrix, with no RHS archive."""
    return SourceSpec(matrix_path=str(spd_matrix_file), rhs_path=None)


@pytest.fixture
def dataset_spec() -> DatasetSpec:
    """Deterministic single-sample dataset spec."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"neutral_ones": 1},
            seed=42,
            shuffle=False,
            strategy_overrides={"neutral_ones": {"samples": 1}},
        ),
        normalize="none",
    )


@pytest.fixture
def make_data_config(spd_matrix_file: Path, tmp_path: Path) -> Callable[..., DataConfigFile]:
    """Factory for a resolved single-matrix `DataConfigFile` with overridable knobs."""

    def _make(
        *,
        dataset_id: str = "ds",
        seed: int = 42,
        matrix_path: Path | None = None,
        data_dir: Path | None = None,
        dataset_format: DatasetFormat = "hdf5",
    ) -> DataConfigFile:
        return DataConfigFile(
            id=dataset_id,
            source=SourceConfig(
                matrix_path=str(matrix_path or spd_matrix_file), problem="Test problem"
            ),
            generation=GenerationConfig(
                normalize="none",
                shuffle=False,
                seed=seed,
                strategy=[StrategyConfig(name="neutral_ones", samples=1)],
            ),
            output=OutputConfig(
                data_dir=data_dir or tmp_path / "out", dataset_format=dataset_format
            ),
        )

    return _make


@pytest.fixture
def data_config(make_data_config: Callable[..., DataConfigFile]) -> DataConfigFile:
    """Default resolved data config."""
    return make_data_config()


@pytest.fixture(params=["hdf5", "zarr"])
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    """Every storage format a dataset can be written in."""
    return request.param


@pytest.fixture
def build_in(
    source_spec: SourceSpec, dataset_spec: DatasetSpec
) -> Callable[[Path, DatasetFormat], Path]:
    """Build the fixture dataset into a directory in the given format."""

    def _build(target: Path, dataset_format: DatasetFormat) -> Path:
        build_test_dataset(source_spec, dataset_spec, str(target), dataset_format=dataset_format)
        return target

    return _build


@pytest.fixture
def generated_dataset_dir(
    tmp_path: Path, build_in: Callable[[Path, DatasetFormat], Path], dataset_format: DatasetFormat
) -> Path:
    """A freshly generated dataset (every generation format), stamped with digests but no identity."""
    return build_in(tmp_path / "dataset", dataset_format)


@pytest.fixture
def solution_block_npy(tmp_path: Path, solution_block: np.ndarray) -> Path:
    """The solution block persisted as one stacked ``.npy`` file."""
    path = tmp_path / "solutions.npy"
    np.save(path, solution_block)
    return path
