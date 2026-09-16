"""Shared fixtures for generation tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from neuralls.domain.generation.interfaces import TracingSolverCallable


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
