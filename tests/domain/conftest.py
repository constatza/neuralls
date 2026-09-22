"""Fixtures for domain-level solver tests."""

from __future__ import annotations

import pytest
import torch
from torchalg.models.result import SolverResult
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    """Seeded well-conditioned SPD matrix (n=20)."""
    generator = torch.Generator().manual_seed(0)
    m = torch.randn(20, 20, dtype=torch.float64, generator=generator)
    return m @ m.T + 20 * torch.eye(20, dtype=torch.float64)


@pytest.fixture
def known_solution() -> torch.Tensor:
    """Seeded ground-truth solution x*."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(20, dtype=torch.float64, generator=generator)


@pytest.fixture
def rhs(spd_matrix: torch.Tensor, known_solution: torch.Tensor) -> torch.Tensor:
    """Right-hand side generated as b = A x*."""
    return spd_matrix @ known_solution


@pytest.fixture
def preconditioners(spd_matrix: torch.Tensor) -> dict[str, object]:
    """Identity and Jacobi preconditioners for the SPD fixture."""
    return {"none": Identity(), "jacobi": JacobiPreconditioner(spd_matrix)}


@pytest.fixture
def perturbed_solution(known_solution: torch.Tensor) -> torch.Tensor:
    """A solution offset from ``known_solution`` by a fixed, known amount."""
    return known_solution + 0.1


@pytest.fixture
def zero_reference_solution() -> torch.Tensor:
    """A zero reference solution, exercising ``relative_exact_error``'s ~0-norm branch."""
    return torch.zeros(20, dtype=torch.float64)


@pytest.fixture
def decreasing_error_history() -> list[float]:
    """A monotonically decreasing error/bound history with a nonzero first entry."""
    return [4.0, 2.0, 1.0, 0.5]


@pytest.fixture
def zero_first_error_history() -> list[float]:
    """An error/bound history whose first entry is zero (undefined normalization)."""
    return [0.0, 1.0, 2.0]


@pytest.fixture
def solver_result_with_exact_energy_history() -> SolverResult:
    """A SolverResult carrying the exact ``error_history_a_norm`` (x_exact was supplied)."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=(4.0, 2.0, 1.0, 0.5),
        energy_decrements=(1.0, 0.5, 0.25),
    )


@pytest.fixture
def solver_result_with_decrements_only() -> SolverResult:
    """A SolverResult with no exact history (no x_exact) but nonempty energy_decrements."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=None,
        energy_decrements=(1.0, 0.5, 0.25),
    )


@pytest.fixture
def solver_result_with_no_energy_data() -> SolverResult:
    """A SolverResult with neither the exact history nor energy_decrements populated."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=None,
        energy_decrements=None,
    )
