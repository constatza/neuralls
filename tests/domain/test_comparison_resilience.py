"""Resilience of the CG comparison runner against reference-solve failures."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torchalg.preconditioners.implementations import CallablePreconditioner

from neuralls.domain.solver import comparison
from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.domain.solver.models.benchmark_events import UnmeasuredResidual

HEALTHY_APPLICATIONS = 3
"""Applications of the poisoned preconditioner that still return a usable direction."""

ITERATION_CAP = 30
"""The solver does not stop on a NaN residual, so a poisoned solve runs to its cap."""


def _poisoned_after(healthy: int, value: float) -> CallablePreconditioner:
    """A preconditioner that returns ``value`` everywhere once ``healthy`` applications ran."""
    applied = 0

    def apply(residual: torch.Tensor) -> torch.Tensor:
        nonlocal applied
        applied += 1
        if applied > healthy:
            return torch.full_like(residual, value)
        return residual * 0.1

    return CallablePreconditioner(apply)


def test_reference_failure_drops_exact_metrics_but_falls_back_to_gm_bound(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """A failing reference solve leaves exact-error metrics unset and still solves,

    falling back to the Golub-Meurant bound for the energy-error signal instead of
    losing it entirely.
    """
    with patch.object(comparison, "reference_solver", side_effect=RuntimeError("too ill")):
        results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    assert all(result.converged for result in results.values())
    assert all(result.exact_error is None for result in results.values())
    assert all(result.error_history_a_rel is None for result in results.values())
    assert all(result.error_bound_a_rel is not None for result in results.values())


def test_energy_error_fields_are_mutually_exclusive(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """Exactly one of error_history_a_rel / error_bound_a_rel is set, never both."""
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    for result in results.values():
        assert (result.error_history_a_rel is None) != (result.error_bound_a_rel is None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_iterate_is_recorded_as_a_failed_solve_not_raised(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, bad: float
) -> None:
    """A learned preconditioner may poison the iterate; the batch must record it and go on."""
    result = run_cg_comparison(
        spd_matrix,
        rhs,
        preconditioners={"p": _poisoned_after(HEALTHY_APPLICATIONS, bad)},
        maxiter=ITERATION_CAP,
    )["p"]

    assert result.converged is False
    assert result.iterations == 0
    assert result.error == f"non-finite iterate after {ITERATION_CAP} iterations"
    assert result.true_residual == UnmeasuredResidual()
    assert result.residual_history_rel == []
    assert result.residual_history_abs == []
    assert result.error_history_a_rel is None
    assert result.error_bound_a_rel is None
    assert result.exact_error is None
    assert np.array_equal(result.x, result.initial_guess)
    assert result.solve_time_seconds is not None
