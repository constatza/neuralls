"""Shared fixtures for generation tests."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from loguru import logger

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.shared.types import DatasetFormat, MatrixFormat


@pytest.fixture
def residual_solver() -> TracingSolverCallable:
    """Default tracing solver for residuals / gaussian_residuals strategies."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def solver_overrides(residual_solver: TracingSolverCallable) -> dict[str, Any]:
    """Solver overrides covering every single-RHS (trace) strategy."""
    return {
        "residuals": residual_solver,
        "gaussian_residuals": residual_solver,
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
        dataset_format: DatasetFormat = "hdf5",
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
        build_dataset(source_spec, dataset_spec, str(target), dataset_format=dataset_format)
        return target

    return _build


@pytest.fixture
def generated_dataset_dir(
    tmp_path: Path, build_in: Callable[[Path, DatasetFormat], Path], dataset_format: DatasetFormat
) -> Path:
    """A freshly generated dataset (every generation format), stamped with digests but no identity."""
    return build_in(tmp_path / "dataset", dataset_format)


# --- archive allocation fixtures ------------------------------------------------


@pytest.fixture
def write_solution_files(tmp_path: Path) -> Callable[[int], str]:
    """Write ``n`` sorted solution files and return their glob pattern.

    File ``k`` holds the constant vector ``k + 1`` of length 4, so its position in
    the sorted pool is recoverable from its contents.
    """

    def _write(n_files: int) -> str:
        directory = tmp_path / "archive"
        directory.mkdir(exist_ok=True)
        for idx in range(n_files):
            np.savetxt(directory / f"solution_{idx:03d}.txt", np.full(4, float(idx + 1)))
        return str(directory / "solution_*.txt")

    return _write


@pytest.fixture
def write_rhs_files(tmp_path: Path) -> Callable[[int], str]:
    """Write ``n`` sorted RHS files and return their glob pattern.

    File ``k`` holds the constant vector ``k + 1`` of length 4, so its position in
    the sorted pool is recoverable from its contents.
    """

    def _write(n_files: int) -> str:
        directory = tmp_path / "rhs_archive"
        directory.mkdir(exist_ok=True)
        for idx in range(n_files):
            np.savetxt(directory / f"rhs_{idx:03d}.txt", np.full(4, float(idx + 1)))
        return str(directory / "rhs_*.txt")

    return _write


@pytest.fixture
def make_trace_solver() -> Callable[[Callable[[int], int]], TracingSolverCallable]:
    """Factory for a stub tracing solver whose trajectory length depends on the call.

    ``trace_length(call_idx)`` gives the number of trajectory rows (iterations 0..k)
    the stub records for the ``call_idx``-th base system. Row ``k`` holds the value
    ``k + 1`` so the recorded rows can be checked. The solver returns a zero solution
    and an info object carrying the residual and solution trajectories as tensors,
    the attributes the residuals strategy reads.
    """

    def _build(trace_length: Callable[[int], int]) -> TracingSolverCallable:
        calls = itertools.count()

        def _solve(
            matrix: np.ndarray,
            rhs: np.ndarray,
            x0: np.ndarray,
            *,
            maxiter: int,
            rtol: float,
            atol: float,
        ) -> tuple[np.ndarray, SimpleNamespace]:
            del rhs, maxiter, rtol, atol
            length = trace_length(next(calls))
            n = matrix.shape[0]
            values = torch.arange(1, length + 1, dtype=torch.float64).reshape(length, 1)
            trace = values.expand(length, n).contiguous()
            info = SimpleNamespace(residual_vectors=trace, solution_vectors=trace)
            return x0, info

        return cast(TracingSolverCallable, _solve)

    return _build


# --- binding-stream fixtures (multi-matrix, explicit solution files) -----------

BINDING_SEED = 7
BINDING_N_UNKNOWNS = 4
BINDING_SOLUTION_ROWS = 4
BINDING_SOURCE_REGEX = r"(\d+)(?!.*\d)"


@pytest.fixture
def three_spd_matrix_dir(tmp_path: Path) -> Path:
    """Three seeded SPD matrices ``A_0.txt``..``A_2.txt``; returns their directory."""
    rng = np.random.default_rng(BINDING_SEED)
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    for i in range(3):
        base = rng.standard_normal((BINDING_N_UNKNOWNS, BINDING_N_UNKNOWNS))
        matrix = base @ base.T + BINDING_N_UNKNOWNS * np.eye(BINDING_N_UNKNOWNS)
        np.savetxt(mat_dir / f"A_{i}.txt", matrix)
    return mat_dir


@pytest.fixture
def solution_block() -> np.ndarray:
    """Seeded ``(BINDING_SOLUTION_ROWS, BINDING_N_UNKNOWNS)`` block of distinct solutions."""
    rng = np.random.default_rng(BINDING_SEED + 1)
    return rng.standard_normal((BINDING_SOLUTION_ROWS, BINDING_N_UNKNOWNS))


@pytest.fixture
def solution_block_npy(tmp_path: Path, solution_block: np.ndarray) -> Path:
    """The solution block persisted as one stacked ``.npy`` file."""
    path = tmp_path / "solutions.npy"
    np.save(path, solution_block)
    return path


@pytest.fixture
def open_binding_stream() -> Callable[..., BatchStream]:
    """Open a batch stream from a source and per-strategy counts, with a fixed seed."""

    def _open(
        source: SourceSpec,
        counts: dict[str, int],
        *,
        overrides: dict[str, dict[str, Any]] | None = None,
        batch_size: int = 1 << 20,
        matrix_format: MatrixFormat = MatrixFormat.DENSE,
    ) -> BatchStream:
        spec = DatasetSpec(
            mixture=MixtureSpec(
                counts=counts,
                seed=BINDING_SEED,
                shuffle=False,
                strategy_overrides=overrides,
            ),
        )
        return open_batch_stream(source, spec, batch_size=batch_size, matrix_format=matrix_format)

    return _open


@pytest.fixture
def two_matrix_source(three_spd_matrix_dir: Path) -> SourceSpec:
    """Source reading the first two matrices of the shared matrix directory as two bindings."""
    return SourceSpec(
        matrix_path=str(three_spd_matrix_dir / "A_[01].txt"),
        sample_id_regex=BINDING_SOURCE_REGEX,
    )


@pytest.fixture
def warning_messages() -> Iterator[list[str]]:
    """Collect loguru WARNING messages emitted during one test."""
    messages: list[str] = []
    sink_id = logger.add(lambda record: messages.append(record.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)
