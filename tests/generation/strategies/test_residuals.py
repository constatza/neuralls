"""Thorough tests for residual-error trace strategies.

Mathematical properties under test:
    - e_k = x_true - x_k  (error = true solution minus current iterate)
    - r_k = b - A @ x_k   (CG residual definition)
    - r_k = A @ e_k        (holds exactly when b = A @ x_true)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.helpers import trace_rows_per_system
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.step_window import StepWindow
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import error_trace_rows

from .conftest import _SolverCallRecorder

ARCHIVE_DIR = Path("archive")


def _expected_trace_systems(samples: int, rows_per_system: int) -> int:
    return max(1, math.ceil(samples / rows_per_system))


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector b for the SPD system."""
    rng = np.random.default_rng(17)
    return rng.standard_normal(spd_matrix.shape[0])


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def solution_files(
    archive_source: InMemoryArchiveSource, spd_matrix: np.ndarray
) -> tuple[list[Path], str]:
    """Register random solution vectors in memory and return (files, glob_pattern).

    Each stored x is a random vector; RHS b = A @ x is computed by the strategy.
    """
    rng = np.random.default_rng(31)
    n = spd_matrix.shape[0]
    files = [
        archive_source.add(ARCHIVE_DIR / f"sol_{i:03d}.txt", rng.standard_normal(n))
        for i in range(4)
    ]
    return files, str(ARCHIVE_DIR / "sol_*.txt")


# ---------------------------------------------------------------------------
# Shapes – single-RHS mode
# ---------------------------------------------------------------------------


def test_residuals_single_rhs_shapes(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """Single-RHS mode: all output arrays have the correct shapes."""
    n = spd_matrix.shape[0]
    requested_rows = 10
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 1, "seed": 0}

    result = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )
    assert result.solutions is not None
    assert result.rhs is not None

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=1))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs.shape == (expected_systems, n)
    assert result.solutions.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None

    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)
    assert et.sample_indices.shape == (total,)
    assert et.iteration_indices.shape == (total,)


# ---------------------------------------------------------------------------
# Shapes – multiple-RHS mode
# ---------------------------------------------------------------------------


def test_residuals_multi_rhs_shapes(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """Multiple-RHS mode: all output arrays have correct shapes."""
    _files, glob_pattern = solution_files
    n = spd_matrix.shape[0]
    requested_rows = 8
    stop = 3
    cfg = {
        "samples": requested_rows,
        "stop": stop,
        "start": 1,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = error_trace_rows(
        run_generation(
            "residuals", spd_matrix, cfg=cfg, solver=residual_solver, archive_source=archive_source
        )
    )
    assert result.rhs is not None

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=1))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None

    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


def test_gaussian_residuals_multi_rhs_shapes(
    spd_matrix: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """Gaussian residuals do not require archive solutions or solution globs."""
    n = spd_matrix.shape[0]
    requested_rows = 8
    stop = 3
    cfg = {"samples": requested_rows, "stop": stop, "start": 1, "seed": 0}

    result = error_trace_rows(
        run_generation(
            "gaussian_residuals",
            spd_matrix,
            cfg=cfg,
            solver=residual_solver,
            archive_source=archive_source,
        )
    )
    assert result.rhs is not None

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=1))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None
    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "options"),
    [
        pytest.param("residuals", {"samples": 10, "stop": 4}, id="archive"),
        pytest.param("residuals", {"samples": 6, "stop": 5, "step": 2}, id="archive-stepped"),
        pytest.param("gaussian_residuals", {"samples": 10, "stop": 4}, id="gaussian"),
    ],
)
def test_trace_pairs_satisfy_the_error_and_residual_identities(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
    strategy: str,
    options: dict[str, int],
) -> None:
    """e_k = x_true - x_k, r_k = b - A x_k and r_k = A e_k hold for every recorded trace pair."""
    _files, glob_pattern = solution_files
    cfg: dict[str, object] = {"start": 1, "seed": 0, **options}
    if strategy == "residuals":
        cfg["solutions_glob"] = glob_pattern

    result = error_trace_rows(
        run_generation(
            strategy, spd_matrix, cfg=cfg, solver=residual_solver, archive_source=archive_source
        )
    )

    et = result.error_traces
    assert et is not None
    assert result.rhs is not None
    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], et.true_solutions[s_idx] - x_k, rtol=1e-12)
        np.testing.assert_allclose(
            et.residuals[i], result.rhs[s_idx] - spd_matrix @ x_k, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(et.residuals[i], spd_matrix @ et.errors[i], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Trace count and indices
# ---------------------------------------------------------------------------


def test_residuals_trace_count(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """Positive samples are exact final trace rows, trimming partial final systems."""
    requested_rows = 7
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 1, "seed": 0}

    result = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )
    assert result.rhs is not None

    et = result.error_traces
    assert et is not None
    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=1))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    assert et.residuals.shape[0] == requested_rows
    assert result.rhs.shape[0] == expected_systems


def test_residuals_sample_indices(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """sample_indices correctly identifies which base system each trace pair belongs to."""
    requested_rows = 15
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 1, "seed": 0}

    result = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )

    et = result.error_traces
    assert et is not None

    entries_per_system = trace_rows_per_system(StepWindow(stop=stop, start=1))
    expected_systems = _expected_trace_systems(requested_rows, entries_per_system)
    for s in range(expected_systems):
        start = s * entries_per_system
        end = min(start + entries_per_system, requested_rows)
        np.testing.assert_array_equal(et.sample_indices[start:end], s)


@pytest.mark.parametrize(
    ("samples", "window", "expected"),
    [
        pytest.param(
            10,
            {"stop": 4, "start": 1},
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            id="every-iterate-from-one",
        ),
        pytest.param(3, {"stop": 5, "start": 1, "step": 2}, [1, 3, 5], id="stride-two"),
        pytest.param(
            3, {"stop": 8, "start": 7}, [7, 8, 7], id="start-near-stop-trims-the-last-system"
        ),
    ],
)
def test_residuals_iteration_indices(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
    samples: int,
    window: dict[str, int],
    expected: list[int],
) -> None:
    """Each system keeps the window's iterates (iterate 0, the base pair, never); the last is trimmed."""
    result = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg={"samples": samples, "seed": 0, **window},
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )

    et = result.error_traces
    assert et is not None
    np.testing.assert_array_equal(et.iteration_indices, np.array(expected, dtype=np.int64))


# ---------------------------------------------------------------------------
# step downsampling ([m,n) range, first-n, last-n)
# ---------------------------------------------------------------------------


def test_residuals_step_reduces_count(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """step=2 yields half the trace entries (stop=5, start=1 -> 5 residuals; 1,3,5 kept as 3).

    Full: 12 requested over ceil(12 / 5) = 3 systems, trimmed to 12 rows.
    Half: 6 requested over ceil(6 / 3) = 2 systems of 3 rows = 6 rows, and 12 // 2 = 6.
    """
    stop = 5
    cfg_full = {"samples": 12, "stop": stop, "start": 1, "seed": 0, "step": 1}
    cfg_half = {"samples": 6, "stop": stop, "start": 1, "seed": 0, "step": 2}

    r_full = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg_full,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )
    assert r_full.error_traces is not None
    r_half = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg_half,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )
    assert r_half.error_traces is not None

    assert r_half.error_traces.residuals.shape[0] == r_full.error_traces.residuals.shape[0] // 2


def test_residuals_samples_minus_one_requires_finite_source(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """samples=-1 is rejected for single-RHS mode because the source is unbounded."""
    with pytest.raises(ValueError):
        error_trace_rows(
            run_generation(
                "residuals",
                spd_matrix,
                cfg={"samples": -1, "stop": 4, "start": 1, "seed": 0},
                solver=residual_solver,
                single_rhs=single_rhs,
                archive_source=archive_source,
            )
        )


# ---------------------------------------------------------------------------
# CPU-waste guarantee: solver is never asked to run past `stop`
# ---------------------------------------------------------------------------


def test_residuals_maxiter_equals_window_stop(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    spy_solver: tuple[TracingSolverCallable, _SolverCallRecorder],
    archive_source: InMemoryArchiveSource,
) -> None:
    """The solver is always called with maxiter == config.window.stop, never more."""
    solver, recorder = spy_solver
    stop = 9
    cfg = {"samples": 4, "stop": stop, "start": 1, "seed": 0}

    error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg,
            solver=solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )

    assert recorder.calls, "solver was never called"
    assert all(m == stop for m in recorder.maxiters())


def test_single_rhs_true_solution_is_the_same_for_csr_and_dense(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> None:
    """The exact solution used as the error target must not depend on the storage format."""
    cfg = {"samples": 2, "stop": 3, "start": 1, "seed": 0}

    dense = error_trace_rows(
        run_generation(
            "residuals",
            spd_matrix,
            cfg=cfg,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )
    sparse = error_trace_rows(
        run_generation(
            "residuals",
            csr_array(spd_matrix),
            cfg=cfg,
            solver=residual_solver,
            single_rhs=single_rhs,
            archive_source=archive_source,
        )
    )

    assert dense.error_traces is not None
    assert sparse.error_traces is not None
    np.testing.assert_allclose(
        sparse.error_traces.true_solutions,
        dense.error_traces.true_solutions,
        rtol=1e-10,
        atol=1e-12,
    )
