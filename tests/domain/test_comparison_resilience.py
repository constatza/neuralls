"""Resilience of the CG comparison runner against reference-solve failures."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import CallablePreconditioner

from neuralls.domain.solver import comparison
from neuralls.domain.solver.models.benchmark_events import UnmeasuredResidual
from neuralls.domain.solver.models.energy_error import (
    EnergyErrorBound,
    EnergyErrorFailed,
    EnergyErrorUnavailable,
    ExactEnergyError,
)
from tests.solver_support import solve_against_reference

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
    preconditioners: dict[str, Preconditioner],
) -> None:
    """A failing reference solve leaves exact-error metrics unset and still solves,

    falling back to the Golub-Meurant bound for the energy-error signal instead of
    losing it entirely.
    """
    with patch.object(comparison, "reference_solver", side_effect=RuntimeError("too ill")):
        results = [
            solve_against_reference(spd_matrix, rhs, preconditioner)
            for preconditioner in preconditioners.values()
        ]

    assert all(result.converged for result in results)
    assert all(result.exact_error is None for result in results)
    assert all(isinstance(result.energy_error, EnergyErrorBound) for result in results)


def test_a_healthy_reference_gives_the_exact_energy_error(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, Preconditioner],
) -> None:
    """With a reference solution the exact curve is reported, never the bound."""
    for preconditioner in preconditioners.values():
        result = solve_against_reference(spd_matrix, rhs, preconditioner)
        assert isinstance(result.energy_error, ExactEnergyError)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_iterate_is_recorded_as_a_failed_solve_not_raised(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, bad: float
) -> None:
    """A learned preconditioner may poison the iterate; the batch must record it and go on."""
    result = solve_against_reference(
        spd_matrix, rhs, _poisoned_after(HEALTHY_APPLICATIONS, bad), maxiter=ITERATION_CAP
    )

    assert result.converged is False
    assert result.iterations == 0
    assert result.error == f"non-finite iterate after {ITERATION_CAP} iterations"
    assert result.true_residual == UnmeasuredResidual()
    assert result.residual_history_rel == []
    assert result.residual_history_abs == []
    assert result.energy_error == EnergyErrorUnavailable("the solve produced no usable iterate")
    assert result.exact_error is None
    assert np.array_equal(result.x, result.initial_guess)
    assert result.solve_time_seconds is not None


def test_a_failing_energy_metric_keeps_the_solve(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, Preconditioner],
) -> None:
    """A metric that raises is a typed failure on the result, not a lost solve."""
    with patch.object(comparison, "extract_energy_error", side_effect=ValueError("bad trace")):
        result = solve_against_reference(spd_matrix, rhs, preconditioners["jacobi"])

    assert result.converged
    assert result.energy_error == EnergyErrorFailed("bad trace")
