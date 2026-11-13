"""Unit tests for the pure numeric helpers in ``domain.solver.error_metrics``.

No solve, no device placement, no preconditioner — these exercise the metric
math directly against fixture tensors/results.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from scipy.sparse.linalg import splu as scipy_splu
from torchalg.models.result import SolverResult

from neuralls.domain.solver.error_metrics import (
    MIN_SAFE_RELATIVE_RESIDUAL,
    ReferenceSolver,
    ReferenceTarget,
    extract_energy_error,
    normalize_error_history,
    reference_solver,
    reference_target,
    relative_exact_error,
)
from neuralls.domain.solver.models.config import REFERENCE_PRECISION_MARGIN
from neuralls.domain.solver.models.energy_error import (
    EnergyErrorBound,
    EnergyErrorUnavailable,
    ExactEnergyError,
)


def test_relative_exact_error_known_pair(
    known_solution: torch.Tensor, perturbed_solution: torch.Tensor
) -> None:
    """A fixed offset from x_exact yields the expected relative error."""
    expected = float(torch.linalg.vector_norm(perturbed_solution - known_solution)) / float(
        torch.linalg.vector_norm(known_solution)
    )
    assert relative_exact_error(perturbed_solution, known_solution) == pytest.approx(expected)


def test_relative_exact_error_falls_back_to_absolute_for_zero_exact(
    perturbed_solution: torch.Tensor, zero_reference_solution: torch.Tensor
) -> None:
    """A ~0 x_exact falls back to the absolute error instead of dividing by zero."""
    expected = float(torch.linalg.vector_norm(perturbed_solution))
    assert relative_exact_error(perturbed_solution, zero_reference_solution) == pytest.approx(
        expected
    )


def test_normalize_error_history_divides_by_first_entry(
    decreasing_error_history: list[float],
) -> None:
    normalized = normalize_error_history(decreasing_error_history)
    assert normalized is not None
    assert normalized[0] == pytest.approx(1.0)
    assert normalized == [
        pytest.approx(v / decreasing_error_history[0]) for v in decreasing_error_history
    ]


def test_normalize_error_history_none_for_zero_first_entry(
    zero_first_error_history: list[float],
) -> None:
    assert normalize_error_history(zero_first_error_history) is None


def test_normalize_error_history_none_for_empty() -> None:
    assert normalize_error_history([]) is None
    assert normalize_error_history(None) is None


def test_extract_energy_error_prefers_exact_history(
    solver_result_with_exact_energy_history: SolverResult,
) -> None:
    """When error_history_a_norm is populated, it wins over the GM bound."""
    match extract_energy_error(solver_result_with_exact_energy_history):
        case ExactEnergyError(history=history):
            assert history[0] == pytest.approx(1.0)
        case other:
            pytest.fail(f"expected the exact energy error, got {other}")


def test_extract_energy_error_falls_back_to_gm_bound(
    solver_result_with_decrements_only: SolverResult,
) -> None:
    """With no exact history but nonempty energy_decrements, the GM bound fills in."""
    match extract_energy_error(solver_result_with_decrements_only):
        case EnergyErrorBound(bound=bound):
            assert bound[0] == pytest.approx(1.0)
        case other:
            pytest.fail(f"expected the Golub-Meurant bound, got {other}")


def test_extract_energy_error_unavailable_when_no_data(
    solver_result_with_no_energy_data: SolverResult,
) -> None:
    assert extract_energy_error(solver_result_with_no_energy_data) == EnergyErrorUnavailable(
        "the solve recorded no energy history"
    )


def test_reference_precision_margin_is_configurable(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    """A looser margin still meets its own (looser) precision target."""
    rtol, margin = 1e-6, 1e-2
    target = reference_target(rtol, margin)
    x = reference_solver(spd_matrix).solve(rhs, target_rel=target.applied_rel)
    relative_residual = float(
        torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(rhs)
    )
    assert relative_residual <= rtol * margin


@pytest.fixture
def second_rhs(spd_matrix: torch.Tensor) -> torch.Tensor:
    """A second right-hand side of the SPD fixture, from a seeded solution."""
    generator = torch.Generator().manual_seed(2)
    return spd_matrix @ torch.randn(spd_matrix.shape[0], dtype=torch.float64, generator=generator)


@pytest.fixture
def singular_csr_matrix() -> torch.Tensor:
    """A CSR matrix with an all-zero row, which no LU can factor."""
    return torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64).to_sparse_csr()


def test_one_sparse_factorization_serves_every_rhs(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, second_rhs: torch.Tensor
) -> None:
    sparse_matrix = spd_matrix.to_sparse_csr()

    with patch("neuralls.domain.solver.error_metrics.splu", wraps=scipy_splu) as factor:
        solver = reference_solver(sparse_matrix)
        target_rel = reference_target(1e-6, REFERENCE_PRECISION_MARGIN).applied_rel
        solutions = [solver.solve(b, target_rel=target_rel) for b in (rhs, second_rhs)]

    assert factor.call_count == 1
    for b, x in zip((rhs, second_rhs), solutions, strict=True):
        residual = torch.linalg.vector_norm(b - spd_matrix @ x) / torch.linalg.vector_norm(b)
        assert float(residual) <= 1e-6 * REFERENCE_PRECISION_MARGIN


def test_singular_sparse_matrix_fails_to_factor(singular_csr_matrix: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="Reference factorization failed"):
        reference_solver(singular_csr_matrix)


def test_reference_rejects_direct_solve_with_excess_residual(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    zero_solver = ReferenceSolver(spd_matrix, lambda b: b * 0.0)

    with pytest.raises(RuntimeError, match="could not reach relative residual"):
        zero_solver.solve(rhs, target_rel=1e-9)


def test_reference_precision_margin_clamps_to_machine_epsilon(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    """A margin*rtol below machine precision is clamped, not asked for literally."""
    rtol, margin = 1e-14, 1e-4  # requested target ~1e-18, below float64 precision
    target = reference_target(rtol, margin)
    assert target == ReferenceTarget(rtol * margin, MIN_SAFE_RELATIVE_RESIDUAL)
    assert target.is_clamped
    x = reference_solver(spd_matrix).solve(rhs, target_rel=target.applied_rel)
    relative_residual = float(
        torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(rhs)
    )
    assert relative_residual <= MIN_SAFE_RELATIVE_RESIDUAL * 10
