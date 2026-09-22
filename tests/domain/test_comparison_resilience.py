"""Resilience of the CG comparison runner against GPU-memory and reference failures."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torchalg.monitoring import TraceMode

from neuralls.domain.solver import comparison
from neuralls.domain.solver.comparison import run_cg_comparison


@pytest.fixture
def traced_modes() -> list[TraceMode]:
    """Collects the trace mode of every solve attempt."""
    return []


@pytest.fixture
def oom_when_traced(traced_modes: list[TraceMode]):
    """`_solve_one` stand-in that runs out of GPU memory only while tracing."""
    real_solve_one = comparison._solve_one

    def fake(*args: object, trace_mode: TraceMode, **kwargs: object):
        traced_modes.append(trace_mode)
        if trace_mode == TraceMode.FULL:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_solve_one(*args, trace_mode=trace_mode, **kwargs)

    return fake


def test_traced_oom_retries_untraced_and_keeps_result(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
    oom_when_traced,
    traced_modes: list[TraceMode],
) -> None:
    """An OOM while tracing degrades to an untraced solve, not a failed preconditioner."""
    with patch.object(comparison, "_solve_one", oom_when_traced):
        results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    assert traced_modes[:2] == [TraceMode.FULL, TraceMode.DISABLED]
    assert all(result.error is None and result.converged for result in results.values())
    assert all(result.error_history_a_rel is None for result in results.values())


def test_reference_failure_drops_exact_metrics_but_not_the_solve(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """A failing reference solve leaves exact-error metrics unset and still solves."""
    with patch.object(comparison, "reference_solution", side_effect=RuntimeError("too ill")):
        results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    assert all(result.converged for result in results.values())
    assert all(result.exact_error is None for result in results.values())
    assert all(result.error_history_a_rel is None for result in results.values())
