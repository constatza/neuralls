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

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.helpers import trace_rows_per_system
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.step_window import StepWindow
from neuralls.domain.generation.strategy_configs import ResidualErrorConfig, SearchDirectionsConfig

from .conftest import _SolverCallRecorder


def _expected_trace_systems(samples: int, rows_per_system: int) -> int:
    return max(1, math.ceil(samples / rows_per_system))


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector b for the SPD system."""
    rng = np.random.default_rng(17)
    return rng.standard_normal(spd_matrix.shape[0])


@pytest.fixture
def solution_files(tmp_path: Path, spd_matrix: np.ndarray) -> tuple[list[Path], str]:
    """Save random solution vectors to disk and return (files, glob_pattern).

    Each stored x is a random vector; RHS b = A @ x is computed by the strategy.
    """
    rng = np.random.default_rng(31)
    n = spd_matrix.shape[0]
    files = []
    for i in range(4):
        x = rng.standard_normal(n)
        p = tmp_path / f"sol_{i:03d}.txt"
        np.savetxt(p, x)
        files.append(p)
    return files, str(tmp_path / "sol_*.txt")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_residuals_strategy_registered() -> None:
    """The primary residual-error trace strategy is registered as 'residuals'."""
    from neuralls.domain.generation.runner import _registry

    assert "residuals" in _registry._strategies


def test_gaussian_residuals_strategy_registered() -> None:
    """Gaussian residual traces are available as an explicit strategy."""
    from neuralls.domain.generation.runner import _registry

    assert "gaussian_residuals" in _registry._strategies


# ---------------------------------------------------------------------------
# Shapes – single-RHS mode
# ---------------------------------------------------------------------------


def test_residuals_single_rhs_shapes(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Single-RHS mode: all output arrays have the correct shapes."""
    n = spd_matrix.shape[0]
    requested_rows = 10
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 0, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)
    assert result.solutions is not None
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
) -> None:
    """Multiple-RHS mode: all output arrays have correct shapes."""
    _files, glob_pattern = solution_files
    n = spd_matrix.shape[0]
    requested_rows = 8
    stop = 3
    cfg = {
        "samples": requested_rows,
        "stop": stop,
        "start": 0,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None

    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


def test_gaussian_residuals_multi_rhs_shapes(
    spd_matrix: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Gaussian residuals do not require archive solutions or solution globs."""
    n = spd_matrix.shape[0]
    requested_rows = 8
    stop = 3
    cfg = {"samples": requested_rows, "stop": stop, "start": 0, "seed": 0}

    result = run_generation("gaussian_residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    total = requested_rows

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None
    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


def test_gaussian_residuals_math_holds(
    spd_matrix: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Gaussian residuals still satisfy e_k = x_true - x_k and r_k = A @ e_k."""
    cfg = {"samples": 10, "stop": 4, "start": 0, "seed": 0}

    result = run_generation("gaussian_residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    et = result.error_traces
    assert et is not None

    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)
        np.testing.assert_allclose(et.residuals[i], spd_matrix @ et.errors[i], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


def test_error_equals_true_minus_current(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
) -> None:
    """e_k = x_true - x_k holds for every recorded trace pair."""
    _files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "stop": 4,
        "start": 0,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    et = result.error_traces
    assert et is not None
    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)


def test_residual_equals_b_minus_ax(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
) -> None:
    """r_k = b - A @ x_k: CG residual matches b - A @ current iterate."""
    _files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "stop": 4,
        "start": 0,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    et = result.error_traces
    assert et is not None
    assert result.rhs is not None

    for i in range(et.residuals.shape[0]):
        s_idx = et.sample_indices[i]
        b = result.rhs[s_idx]
        x_k = et.solutions_current[i]
        expected_residual = b - spd_matrix @ x_k
        np.testing.assert_allclose(et.residuals[i], expected_residual, rtol=1e-10, atol=1e-10)


def test_residual_equals_a_times_error(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
) -> None:
    """r_k = A @ e_k: holds exactly because b = A @ x_true in archive mode."""
    _files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "stop": 4,
        "start": 0,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    et = result.error_traces
    assert et is not None

    for i in range(et.residuals.shape[0]):
        a_times_e = spd_matrix @ et.errors[i]
        np.testing.assert_allclose(et.residuals[i], a_times_e, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Trace count and indices
# ---------------------------------------------------------------------------


def test_residuals_trace_count(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Positive samples are exact final trace rows, trimming partial final systems."""
    requested_rows = 7
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 0, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    rows_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_systems = _expected_trace_systems(requested_rows, rows_per_system)
    assert et.residuals.shape[0] == requested_rows
    assert result.rhs is not None
    assert result.rhs.shape[0] == expected_systems


def test_residuals_sample_indices(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """sample_indices correctly identifies which base system each trace pair belongs to."""
    requested_rows = 15
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 0, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None

    entries_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_systems = _expected_trace_systems(requested_rows, entries_per_system)
    for s in range(expected_systems):
        start = s * entries_per_system
        end = min(start + entries_per_system, requested_rows)
        np.testing.assert_array_equal(et.sample_indices[start:end], s)


def test_residuals_iteration_indices(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """iteration_indices run 0..stop for each system when start=0, step=1."""
    requested_rows = 10
    stop = 4
    cfg = {"samples": requested_rows, "stop": stop, "start": 0, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None

    entries_per_system = trace_rows_per_system(StepWindow(stop=stop, start=0))
    expected_iters = np.arange(entries_per_system, dtype=np.int64)
    expected_systems = _expected_trace_systems(requested_rows, entries_per_system)
    for s in range(expected_systems):
        start = s * entries_per_system
        end = min(start + entries_per_system, requested_rows)
        np.testing.assert_array_equal(
            et.iteration_indices[start:end], expected_iters[: end - start]
        )


# ---------------------------------------------------------------------------
# step downsampling ([m,n) range, first-n, last-n)
# ---------------------------------------------------------------------------


def test_residuals_step_reduces_count(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """step=2 yields half the trace entries (stop=5 -> 6 residuals, divisible by 2)."""
    stop = 5  # 6 residuals per sample (0..5)
    cfg_full = {"samples": 12, "stop": stop, "start": 0, "seed": 0, "step": 1}
    cfg_half = {"samples": 6, "stop": stop, "start": 0, "seed": 0, "step": 2}

    r_full = run_generation(
        "residuals", spd_matrix, cfg=cfg_full, solver=residual_solver, single_rhs=single_rhs
    )
    r_half = run_generation(
        "residuals", spd_matrix, cfg=cfg_half, solver=residual_solver, single_rhs=single_rhs
    )

    assert r_full.error_traces is not None
    assert r_half.error_traces is not None
    assert r_half.error_traces.residuals.shape[0] == r_full.error_traces.residuals.shape[0] // 2


def test_residuals_step_indices_correct(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """step=2 with stop=5, start=0: iteration_indices are [0, 2, 4] not [0, 1, 2]."""
    cfg = {"samples": 3, "stop": 5, "start": 0, "seed": 0, "step": 2}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    np.testing.assert_array_equal(et.iteration_indices, np.array([0, 2, 4], dtype=np.int64))


def test_residuals_step_math_still_holds(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
    residual_solver: TracingSolverCallable,
) -> None:
    """e_k = x_true - x_k and r_k = A @ e_k hold even after step downsampling."""
    _files, glob_pattern = solution_files
    cfg = {
        "samples": 6,
        "stop": 5,
        "start": 0,
        "seed": 0,
        "solutions_glob": glob_pattern,
        "step": 2,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg, solver=residual_solver)

    et = result.error_traces
    assert et is not None

    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)
        np.testing.assert_allclose(et.residuals[i], spd_matrix @ et.errors[i], rtol=1e-9, atol=1e-9)


def test_residuals_last_n_via_negative_start(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """start=-3 keeps the last 3 iterations of a stop=8 trace: indices [6, 7, 8]."""
    cfg = {"samples": 3, "stop": 8, "start": -3, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    np.testing.assert_array_equal(et.iteration_indices, np.array([6, 7, 8], dtype=np.int64))


def test_residuals_samples_minus_one_requires_finite_source(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """samples=-1 is rejected for single-RHS mode because the source is unbounded."""
    with pytest.raises(ValueError, match="samples=-1"):
        run_generation(
            "residuals",
            spd_matrix,
            cfg={"samples": -1, "stop": 4, "start": 0, "seed": 0},
            solver=residual_solver,
            single_rhs=single_rhs,
        )


# ---------------------------------------------------------------------------
# Required `stop`, default (last-only)
# ---------------------------------------------------------------------------


def test_residuals_requires_stop(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Omitting the required `stop` field must raise, not run with a silent default."""
    with pytest.raises(Exception, match="stop"):
        run_generation(
            "residuals",
            spd_matrix,
            cfg={"samples": 3, "seed": 0},
            solver=residual_solver,
            single_rhs=single_rhs,
        )


def test_residuals_default_start_keeps_only_last_iterate(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """With `start` unset, every kept row is the final (stop-th) CG iterate — one row per system."""
    stop = 6
    cfg = {"samples": 3, "stop": stop, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    assert et.residuals.shape == (3, spd_matrix.shape[0])
    np.testing.assert_array_equal(et.iteration_indices, np.full(3, stop, dtype=np.int64))


# ---------------------------------------------------------------------------
# CPU-waste guarantee: solver is never asked to run past `stop`
# ---------------------------------------------------------------------------


def test_residuals_maxiter_equals_window_stop(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    spy_solver: tuple[TracingSolverCallable, _SolverCallRecorder],
) -> None:
    """The solver is always called with maxiter == config.window.stop, never more."""
    solver, recorder = spy_solver
    stop = 9
    cfg = {"samples": 4, "stop": stop, "start": 0, "seed": 0}

    run_generation("residuals", spd_matrix, cfg=cfg, solver=solver, single_rhs=single_rhs)

    assert recorder.calls, "solver was never called"
    assert all(m == stop for m in recorder.maxiters())


# ---------------------------------------------------------------------------
# Real convergence tolerance
# ---------------------------------------------------------------------------


def test_residuals_real_tolerance_stops_before_stop(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """A real, reachable rtol lets CG converge well before the `stop` safety cap.

    `start` unset (default: keep only the true last iterate) correctly
    resolves to wherever CG actually stopped, not to a hardcoded index.
    """
    cfg = {"samples": 3, "stop": 200, "rtol": 1e-6, "atol": 1e-10, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    assert np.all(et.iteration_indices < 200)
    assert np.all(et.iteration_indices == et.iteration_indices[0])  # same true stop each run


def test_residuals_real_tolerance_with_forward_start_rejected(
    spd_matrix: np.ndarray,
) -> None:
    """A real tolerance combined with a non-negative (absolute forward) start is rejected.

    SOLID review finding 3: this combination can silently select zero rows
    if CG converges before reaching `start` — reject it outright instead.
    """
    with pytest.raises(Exception, match="rtol"):
        ResidualErrorConfig(samples=3, stop=50, start=45, rtol=1e-6)


def test_residuals_real_tolerance_with_end_relative_start_ok() -> None:
    """A real tolerance combined with an end-relative (negative) start is accepted."""
    config = ResidualErrorConfig(samples=3, stop=50, start=-3, rtol=1e-6)
    assert config.window.start == -3


def test_residuals_real_tolerance_with_default_start_ok() -> None:
    """A real tolerance with `start` left unset (last-only) is accepted."""
    config = ResidualErrorConfig(samples=3, stop=50, rtol=1e-6)
    assert config.window.start is None


# ---------------------------------------------------------------------------
# SOLID review finding 1: SearchDirectionsConfig no longer exposes archive fields
# ---------------------------------------------------------------------------


def test_search_directions_config_rejects_archive_fields() -> None:
    """SearchDirectionsConfig has no archive-related fields (ISP: it never uses them)."""
    with pytest.raises(Exception, match="solutions_glob"):
        SearchDirectionsConfig(samples=3, stop=5, solutions_glob="*.npy")


def test_residual_error_config_still_accepts_archive_fields() -> None:
    """ResidualErrorConfig (residuals.py's strategies) still accepts archive fields."""
    config = ResidualErrorConfig(samples=3, stop=5, solutions_glob="*.npy")
    assert config.solutions_glob == "*.npy"
