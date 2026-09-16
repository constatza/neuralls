"""Tests for SearchDirectionsStrategy (search_directions)."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.runner import SingleRhsStrategyRegistration, _registry
from neuralls.domain.generation.strategy_configs import SearchDirectionsConfig

from .conftest import _SolverCallRecorder


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector for the SPD system."""
    rng = np.random.default_rng(55)
    return rng.standard_normal(spd_matrix.shape[0])


def test_search_directions_registered_as_single_rhs_strategy() -> None:
    """SearchDirectionsStrategy is registered with explicit single-RHS capability."""
    registration = _registry._strategies["search_directions"]
    assert isinstance(registration, SingleRhsStrategyRegistration)


def test_search_directions_config_accepts_step(
    spd_matrix: np.ndarray, direction_solver: TracingSolverCallable
) -> None:
    """Config accepts `step` and yields traced direction/product pairs."""
    cfg = {"samples": 2, "stop": 4, "start": 0, "seed": 0, "step": 2}

    result = run_generation(
        "search_directions",
        spd_matrix,
        cfg=cfg,
        solver=direction_solver,
        single_rhs=np.ones(spd_matrix.shape[0]),
    )

    assert result.residual_traces is not None
    assert result.residual_traces.search_directions is not None
    assert result.residual_traces.search_direction_products is not None
    assert (
        result.residual_traces.search_directions.shape
        == result.residual_traces.search_direction_products.shape
    )
    assert result.residual_traces.search_directions.shape[0] > 0


def test_search_directions_collects_repeated_single_rhs(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    direction_solver: TracingSolverCallable,
) -> None:
    """Single-RHS mode keeps the same RHS while collecting CG search directions."""
    cfg = {"samples": 3, "stop": 4, "start": 0, "seed": 0}

    result = run_generation(
        "search_directions", spd_matrix, cfg=cfg, solver=direction_solver, single_rhs=single_rhs
    )

    assert result.rhs is not None
    assert result.rhs.shape[1] == single_rhs.shape[0]
    for row in result.rhs:
        np.testing.assert_allclose(row, single_rhs)

    assert result.residual_traces is not None
    assert result.residual_traces.search_directions is not None
    assert result.residual_traces.search_direction_products is not None


def test_search_direction_products_match_matrix_action(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    direction_solver: TracingSolverCallable,
) -> None:
    """Stored products equal `A @ p_k` for the traced directions."""
    cfg = {"samples": 1, "stop": 3, "start": 0, "seed": 0}

    result = run_generation(
        "search_directions", spd_matrix, cfg=cfg, solver=direction_solver, single_rhs=single_rhs
    )
    traces = result.residual_traces

    assert traces is not None
    assert traces.search_directions is not None
    assert traces.search_direction_products is not None

    expected = np.array([spd_matrix @ direction for direction in traces.search_directions])
    np.testing.assert_allclose(traces.search_direction_products, expected)


def test_search_directions_default_start_keeps_only_last(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, direction_solver: TracingSolverCallable
) -> None:
    """With `start` unset, every kept row is the final (stop-th) direction — one per system."""
    stop = 5
    cfg = {"samples": 3, "stop": stop, "seed": 0}

    result = run_generation(
        "search_directions", spd_matrix, cfg=cfg, solver=direction_solver, single_rhs=single_rhs
    )

    traces = result.residual_traces
    assert traces is not None
    assert traces.solutions.shape == (3, spd_matrix.shape[0])
    np.testing.assert_array_equal(traces.iteration_indices, np.full(3, stop, dtype=np.int64))


def test_search_directions_last_n_via_negative_start(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, direction_solver: TracingSolverCallable
) -> None:
    """start=-2 keeps the last 2 iterations of a stop=6 trace: indices [5, 6]."""
    cfg = {"samples": 2, "stop": 6, "start": -2, "seed": 0}

    result = run_generation(
        "search_directions", spd_matrix, cfg=cfg, solver=direction_solver, single_rhs=single_rhs
    )

    traces = result.residual_traces
    assert traces is not None
    np.testing.assert_array_equal(traces.iteration_indices, np.array([5, 6], dtype=np.int64))


def test_search_directions_requires_stop(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, direction_solver: TracingSolverCallable
) -> None:
    """Omitting the required `stop` field must raise, not run with a silent default."""
    with pytest.raises(Exception, match="stop"):
        run_generation(
            "search_directions",
            spd_matrix,
            cfg={"samples": 2, "seed": 0},
            solver=direction_solver,
            single_rhs=single_rhs,
        )


def test_search_directions_maxiter_equals_window_stop(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
    spy_solver: tuple[TracingSolverCallable, _SolverCallRecorder],
) -> None:
    """The solver is always called with maxiter == config.window.stop, never more."""
    solver, recorder = spy_solver
    stop = 7
    cfg = {"samples": 3, "stop": stop, "start": 0, "seed": 0}

    run_generation("search_directions", spd_matrix, cfg=cfg, solver=solver, single_rhs=single_rhs)

    assert recorder.calls, "solver was never called"
    assert all(m == stop for m in recorder.maxiters())


def test_search_directions_real_tolerance_stops_before_stop(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, direction_solver: TracingSolverCallable
) -> None:
    """A real, reachable rtol lets CG converge well before the `stop` safety cap."""
    cfg = {"samples": 2, "stop": 200, "rtol": 1e-6, "atol": 1e-10, "seed": 0}

    result = run_generation(
        "search_directions", spd_matrix, cfg=cfg, solver=direction_solver, single_rhs=single_rhs
    )

    traces = result.residual_traces
    assert traces is not None
    assert np.all(traces.iteration_indices < 200)


def test_search_directions_real_tolerance_with_forward_start_rejected() -> None:
    """SOLID review finding 3 applies to SearchDirectionsConfig too (shares `_CgTraceFields`)."""
    with pytest.raises(Exception, match="rtol"):
        SearchDirectionsConfig(samples=2, stop=50, start=45, rtol=1e-6)
