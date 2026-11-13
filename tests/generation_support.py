"""Generation fixtures and helpers shared by ``tests/generation`` and ``tests/io/generation``.

Each conftest re-exports the fixtures its tests use, so one definition serves both
directories (the same pattern as ``tests/atomic_support.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from loguru import logger

from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.interfaces import (
    ErrorTraceRows,
    GeneratedSamples,
    ResidualTraceRows,
    SolvedRows,
    TracingSolverCallable,
)
from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.archive_source import FsArchiveSource
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix
from neuralls.shared.types import MatrixFormat

ARCHIVE_SOURCE = FsArchiveSource()

_READERS = MatrixReaders(
    generic=partial(read_matrix, lazy=True), dense=partial(read_dense_npy, lazy=True)
)

BINDING_SEED = 7
BINDING_N_UNKNOWNS = 4
BINDING_MATRIX_COUNT = 3
BINDING_SOLUTION_ROWS = 4
BINDING_SOURCE_REGEX = r"(\d+)(?!.*\d)"


SUBDOMAIN_PARAMETER_SETS = [
    (10554158, 20662154, 19907116, 20715669),
    (10846921, 25611823, 31377296, 28205425),
    (12000001, 22000002, 32000003, 42000004),
]


def spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random symmetric positive-definite ``n``×``n`` matrix."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def binding_matrices() -> list[np.ndarray]:
    """Seeded SPD matrices, one per binding, shared by the on-disk and in-memory sources."""
    rng = np.random.default_rng(BINDING_SEED)
    matrices = []
    for _ in range(BINDING_MATRIX_COUNT):
        base = rng.standard_normal((BINDING_N_UNKNOWNS, BINDING_N_UNKNOWNS))
        matrices.append(base @ base.T + BINDING_N_UNKNOWNS * np.eye(BINDING_N_UNKNOWNS))
    return matrices


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
def warning_messages() -> Iterator[list[str]]:
    """Collect loguru WARNING messages emitted during one test."""
    messages: list[str] = []
    sink_id = logger.add(lambda record: messages.append(record.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def subdomain_matrices() -> dict[str, np.ndarray]:
    """SPD matrices by filename whose stems would collide under the default trailing-digit regex."""
    rng = np.random.default_rng(123)
    return {
        f"E1_{e1}_E2_{e2}_E3_{e3}_E4_{e4}_subdomain_1_Kaa.txt": spd_matrix(rng, 4)
        for e1, e2, e3, e4 in SUBDOMAIN_PARAMETER_SETS
    }


@pytest.fixture
def three_spd_matrix_dir(tmp_path: Path) -> Path:
    """Three seeded SPD matrices ``A_0.txt``..``A_2.txt``; returns their directory."""
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    for i, matrix in enumerate(binding_matrices()):
        np.savetxt(mat_dir / f"A_{i}.txt", matrix)
    return mat_dir


@pytest.fixture
def solution_block() -> np.ndarray:
    """Seeded ``(BINDING_SOLUTION_ROWS, BINDING_N_UNKNOWNS)`` block of distinct solutions."""
    rng = np.random.default_rng(BINDING_SEED + 1)
    return rng.standard_normal((BINDING_SOLUTION_ROWS, BINDING_N_UNKNOWNS))


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
        return open_batch_stream(
            source,
            spec,
            batch_size=batch_size,
            matrix_format=matrix_format,
            readers=_READERS,
            archive_source=ARCHIVE_SOURCE,
        )

    return _open


def solved_rows(result: GeneratedSamples) -> SolvedRows:
    """The result of a strategy that emits (rhs, solution) pairs; any other shape fails."""
    match result:
        case SolvedRows():
            return result
        case ResidualTraceRows() | ErrorTraceRows():
            raise TypeError(f"Expected SolvedRows, got {type(result).__name__}")


def error_trace_rows(result: GeneratedSamples) -> ErrorTraceRows:
    """The result of a CG trajectory strategy; any other shape fails."""
    match result:
        case ErrorTraceRows():
            return result
        case SolvedRows() | ResidualTraceRows():
            raise TypeError(f"Expected ErrorTraceRows, got {type(result).__name__}")


def residual_trace_rows(result: GeneratedSamples) -> ResidualTraceRows:
    """The result of a smoother-probe strategy; any other shape fails."""
    match result:
        case ResidualTraceRows():
            return result
        case SolvedRows() | ErrorTraceRows():
            raise TypeError(f"Expected ResidualTraceRows, got {type(result).__name__}")
