"""Trajectory rows per base system: full expansion is kept, early convergence is an error.

The window is 0..3, so K_rows = 4 and a base system that runs to ``stop`` yields 4 rows.
The stub solver in ``conftest.py`` records a trajectory of a length chosen per call.
A single shared RHS is used so the strategy needs no archive: each base system is one solve.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.runner import run_generation
from neuralls.domain.generation.trace_utils import TrajectoryShortfallError

WINDOW_STOP = 3
K_ROWS = WINDOW_STOP + 1
WINDOW_CFG: dict[str, int] = {"start": 0, "stop": WINDOW_STOP}


def _residuals_cfg(samples: int) -> dict[str, int]:
    return {**WINDOW_CFG, "samples": samples}


def test_trajectory_expansion_drops_no_rows_per_base_system(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    make_trace_solver: Callable[[Callable[[int], int]], TracingSolverCallable],
) -> None:
    """R = 10 rows: B = ceil(10 / 4) = 3 base systems of 4 rows, trimmed to 10.

    Only the last base system is trimmed (4 + 4 + 2 = 10), so each full trajectory
    contributes all of its K_rows rows.
    """
    solver = make_trace_solver(lambda _call: K_ROWS)

    generated = run_generation(
        "residuals",
        small_spd_matrix,
        cfg=_residuals_cfg(10),
        solver=solver,
        single_rhs=small_rhs,
    )

    assert generated.error_traces is not None
    per_system = np.bincount(generated.error_traces.sample_indices)
    np.testing.assert_array_equal(per_system, [K_ROWS, K_ROWS, 2])
    assert generated.error_traces.residuals.shape[0] == 10


def test_early_converging_base_system_raises_trajectory_shortfall(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    make_trace_solver: Callable[[Callable[[int], int]], TracingSolverCallable],
) -> None:
    """Base system 0 records 2 of its 4 rows: an error, with no padded output."""
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
    assert "produced 2 trace rows" in message
    assert "expected 4" in message
    assert "converged before window.stop=3" in message
