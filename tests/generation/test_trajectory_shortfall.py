"""Trajectory rows per base system: full expansion is kept, early convergence is an error.

The window is 1..3, so K_rows = stop - start + 1 = 3. A base system that runs to
``stop`` records a full trajectory of stop + 1 = 4 rows (iterates 0..3), and the
window keeps iterates 1..3, i.e. 3 rows.
The stub solver in ``conftest.py`` records a trajectory of a length chosen per call.
A single shared RHS is used so the strategy needs no archive: each base system is one solve.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.runner import run_generation
from neuralls.domain.generation.trace_utils import TrajectoryShortfallError

WINDOW_START = 1
WINDOW_STOP = 3
K_ROWS = WINDOW_STOP - WINDOW_START + 1
FULL_TRAJECTORY_LENGTH = WINDOW_STOP + 1
WINDOW_CFG: dict[str, int] = {"start": WINDOW_START, "stop": WINDOW_STOP}


def _residuals_cfg(samples: int) -> dict[str, int]:
    return {**WINDOW_CFG, "samples": samples}


def test_trajectory_expansion_drops_no_rows_per_base_system(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    make_trace_solver: Callable[[Callable[[int], int]], TracingSolverCallable],
) -> None:
    """R = 10 rows: B = ceil(10 / 3) = 4 base systems of K_rows = 3 rows, trimmed to 10.

    Only the last base system is trimmed (3 + 3 + 3 + 1 = 10), so each full
    trajectory contributes all of its K_rows rows.
    """
    solver = make_trace_solver(lambda _call: FULL_TRAJECTORY_LENGTH)

    generated = run_generation(
        "residuals",
        small_spd_matrix,
        cfg=_residuals_cfg(10),
        solver=solver,
        single_rhs=small_rhs,
    )
    assert generated.error_traces is not None
    assert generated.error_traces.residuals is not None
    assert generated.error_traces is not None

    per_system = np.bincount(generated.error_traces.sample_indices)
    np.testing.assert_array_equal(per_system, [K_ROWS, K_ROWS, K_ROWS, 1])
    assert generated.error_traces.residuals.shape[0] == 10


def test_early_converging_base_system_raises_trajectory_shortfall(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    make_trace_solver: Callable[[Callable[[int], int]], TracingSolverCallable],
) -> None:
    """Base system 0 records a 2-row trajectory (iterates 0..1): the window keeps 1 row, not 3.

    Expected K_rows = 3, so the shortfall message reads "produced 1", "expected 3".
    """
    solver = make_trace_solver(lambda _call: 2)

    with pytest.raises(TrajectoryShortfallError) as excinfo:
        run_generation(
            "residuals",
            small_spd_matrix,
            cfg=_residuals_cfg(4),
            solver=solver,
            single_rhs=small_rhs,
        )

    message = str(excinfo.value)
    assert "base system 0" in message
    assert "produced 1 trace rows" in message
    assert "expected 3" in message
    assert "converged before window.stop=3" in message


def test_real_solver_converging_early_raises_shortfall_naming_the_system() -> None:
    """A = 2I makes CG exact after one step, so a window that keeps steps 1 and 3 falls short.

    The base pair is step 0 and only step 1 is recorded, so the window keeps one row where
    it expects two. No stub solver is involved: the default CG tracer produces the trajectory.
    """
    matrix = 2.0 * np.eye(4)
    rhs = np.ones(4)
    cfg = {"samples": 2, "start": 1, "stop": 4, "step": 2, "seed": 0}

    solver = make_solver()

    with pytest.raises(TrajectoryShortfallError) as excinfo:
        run_generation("residuals", matrix, cfg=cfg, solver=solver, single_rhs=rhs)

    message = str(excinfo.value)
    assert "base system 0" in message
    assert "produced 1 trace rows" in message
    assert "expected 2" in message
