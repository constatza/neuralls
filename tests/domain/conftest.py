"""Fixtures for domain-level solver tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchalg.models.result import SolverResult
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner

from neuralls.domain.solver.models.result import CGComparisonResult, StageCost
from neuralls.shared.types import CostProvenance


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
    """Identity and Jacobi preconditioners for the SPD fixture.

    Keyed "identity", matching `run_cg_comparison`'s own auto-injected
    baseline key — providing it explicitly here means no second, redundant
    baseline solve gets silently added on top of these two.
    """
    return {"identity": Identity(), "jacobi": JacobiPreconditioner(spd_matrix)}


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
def comparison_result_with_cost_data() -> CGComparisonResult:
    """A converged CGComparisonResult with setup/solve time and memory measured."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="jacobi",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=StageCost(
            wall_time_seconds=2.0, peak_memory_bytes=1000, provenance=CostProvenance.MEASURED
        ),
        solve_time_seconds=1.0,
        solve_peak_memory_bytes=2000,
    )


@pytest.fixture
def comparison_result_with_unavailable_setup_cost() -> CGComparisonResult:
    """A converged CGComparisonResult whose setup cost is UNAVAILABLE-provenance.

    Direct regression fixture for the original bug: a silently-failed
    historical lookup falls back to a near-zero measured value that must
    never render identically to a genuine near-zero build.
    """
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="neural",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        generation_cost=None,
        setup_cost=StageCost(
            wall_time_seconds=0.002,
            peak_memory_bytes=None,
            provenance=CostProvenance.UNAVAILABLE,
        ),
        solve_time_seconds=1.0,
        solve_peak_memory_bytes=2000,
    )


@pytest.fixture
def comparison_result_with_generation_cost() -> CGComparisonResult:
    """A converged CGComparisonResult carrying both a historical generation
    cost and a freshly-measured setup cost."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="pod2g",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        generation_cost=StageCost(
            wall_time_seconds=5.0,
            peak_memory_bytes=500,
            provenance=CostProvenance.HISTORICAL,
        ),
        setup_cost=StageCost(
            wall_time_seconds=2.0, peak_memory_bytes=1000, provenance=CostProvenance.MEASURED
        ),
        solve_time_seconds=1.0,
        solve_peak_memory_bytes=2000,
    )


@pytest.fixture
def comparison_result_without_cost_data() -> CGComparisonResult:
    """A converged CGComparisonResult with no cost measurements attached."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="jacobi",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )


@pytest.fixture
def comparison_result_with_zero_iterations() -> CGComparisonResult:
    """A CGComparisonResult where the "identity" baseline converged immediately (0 iterations)."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=0,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1e-8],
        residual_history_abs=[1e-8],
        preconditioner="identity",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        solve_time_seconds=0.0,
        solve_peak_memory_bytes=0,
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
