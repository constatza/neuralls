"""Shared fixtures for generation strategy tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from neuralls.domain.generation.interfaces import TracingSolverCallable


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """10x10 symmetric positive-definite matrix."""
    n = 10
    A = np.random.randn(n, n)
    return A.T @ A + np.eye(n)


@pytest.fixture
def sample_matrix(spd_matrix: np.ndarray) -> np.ndarray:
    """Alias for spd_matrix, used in validated_archive tests."""
    return spd_matrix


@pytest.fixture
def residual_solver() -> TracingSolverCallable:
    """Default tracing solver for residuals / residual_traces strategies."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def direction_solver() -> TracingSolverCallable:
    """Default tracing solver for search_directions strategy."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@dataclass
class _SolverCallRecorder:
    """Records the keyword arguments of every call to a wrapped solver."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def maxiters(self) -> list[int]:
        """The `maxiter` recorded on each call, in call order."""
        return [call["maxiter"] for call in self.calls]


@pytest.fixture
def spy_solver(
    residual_solver: TracingSolverCallable,
) -> tuple[TracingSolverCallable, _SolverCallRecorder]:
    """Wrap `residual_solver`, recording each call's kwargs (notably `maxiter`).

    Used to verify the CPU-waste guarantee directly — that a strategy never
    calls the solver with `maxiter` greater than `StepWindow.stop`.
    """
    recorder = _SolverCallRecorder()

    def wrapped(
        A: NDArray, b: NDArray, x0: NDArray, *, maxiter: int, rtol: float, atol: float
    ) -> tuple[NDArray, Any]:
        recorder.calls.append({"maxiter": maxiter, "rtol": rtol, "atol": atol})
        return residual_solver(A, b, x0, maxiter=maxiter, rtol=rtol, atol=atol)

    return wrapped, recorder
