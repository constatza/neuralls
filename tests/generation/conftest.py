"""Shared fixtures for generation tests."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from neuralls.domain.generation.interfaces import TracingSolverCallable
from tests import generation_support

# Re-exported so these tests share the fixtures with tests/io/generation.
residual_solver = generation_support.residual_solver
solver_overrides = generation_support.solver_overrides
warning_messages = generation_support.warning_messages
solution_block = generation_support.solution_block


@pytest.fixture
def labeled_trajectory() -> Callable[[int], np.ndarray]:
    """Factory building a `(length, 1)` array whose row `i` holds the value `i`.

    Used by `StepWindow` tests to verify selection against arbitrary
    trajectory lengths without needing a real solver/smoother run.
    """

    def _make(length: int) -> np.ndarray:
        return np.arange(length, dtype=np.int64).reshape(length, 1)

    return _make


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
