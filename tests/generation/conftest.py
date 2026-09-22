"""Shared fixtures for generation tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.shared.types import DatasetFormat


@pytest.fixture
def residual_solver() -> TracingSolverCallable:
    """Default tracing solver for residuals / gaussian_residuals strategies."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def direction_solver() -> TracingSolverCallable:
    """Default tracing solver for the search_directions strategy."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def solver_overrides(
    residual_solver: TracingSolverCallable, direction_solver: TracingSolverCallable
) -> dict[str, Any]:
    """Solver overrides covering every single-RHS (trace) strategy."""
    return {
        "residuals": residual_solver,
        "gaussian_residuals": residual_solver,
        "search_directions": direction_solver,
    }


@pytest.fixture
def labeled_trajectory() -> Callable[[int], np.ndarray]:
    """Factory building a `(length, 1)` array whose row `i` holds the value `i`.

    Used by `StepWindow` tests to verify selection against arbitrary
    trajectory lengths without needing a real solver/smoother run.
    """

    def _make(length: int) -> np.ndarray:
        return np.arange(length, dtype=np.int64).reshape(length, 1)

    return _make


# --- identity / reuse fixtures -------------------------------------------------


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
        dataset_format: DatasetFormat = "npy",
    ) -> DataConfigFile:
        return DataConfigFile(
            id=dataset_id,
            source=SourceConfig(matrix_path=str(matrix_path or spd_matrix_file)),
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


@pytest.fixture(params=["npy", "hdf5", "zarr"])
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    """Every storage format a dataset can be written in."""
    return request.param


@pytest.fixture
def build_in(
    source_spec: SourceSpec, dataset_spec: DatasetSpec
) -> Callable[[Path, DatasetFormat], Path]:
    """Build the fixture dataset into a directory in the given format."""

    def _build(target: Path, dataset_format: DatasetFormat) -> Path:
        build_dataset(source_spec, dataset_spec, str(target), dataset_format=dataset_format)
        return target

    return _build


@pytest.fixture
def generated_dataset_dir(
    tmp_path: Path, build_in: Callable[[Path, DatasetFormat], Path], dataset_format: DatasetFormat
) -> Path:
    """A freshly generated dataset (all formats), stamped with digests but no identity."""
    return build_in(tmp_path / "dataset", dataset_format)


@pytest.fixture
def npy_dataset_dir(tmp_path: Path, build_in: Callable[[Path, DatasetFormat], Path]) -> Path:
    """A freshly generated npy dataset, whose artifacts are plain editable files."""
    return build_in(tmp_path / "npy_dataset", "npy")
