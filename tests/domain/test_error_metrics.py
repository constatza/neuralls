"""Unit tests for the pure numeric helpers in ``domain.solver.error_metrics``.

No solve, no device placement, no preconditioner — these exercise the metric
math directly against fixture tensors/results.
"""

from __future__ import annotations

import pytest
import torch
from torchalg.models.result import SolverResult

from neuralls.domain.solver.error_metrics import (
    MIN_SAFE_RELATIVE_RESIDUAL,
    extract_energy_error,
    normalize_error_history,
    reference_solution,
    relative_exact_error,
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
    error_history, error_bound = extract_energy_error(solver_result_with_exact_energy_history)
    assert error_history is not None
    assert error_history[0] == pytest.approx(1.0)
    assert error_bound is None


def test_extract_energy_error_falls_back_to_gm_bound(
    solver_result_with_decrements_only: SolverResult,
) -> None:
    """With no exact history but nonempty energy_decrements, the GM bound fills in."""
    error_history, error_bound = extract_energy_error(solver_result_with_decrements_only)
    assert error_history is None
    assert error_bound is not None
    assert error_bound[0] == pytest.approx(1.0)


def test_extract_energy_error_none_when_no_data(
    solver_result_with_no_energy_data: SolverResult,
) -> None:
    assert extract_energy_error(solver_result_with_no_energy_data) == (None, None)


def test_reference_precision_margin_is_configurable(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    """A looser margin still meets its own (looser) precision target."""
    rtol, margin = 1e-6, 1e-2
    x = reference_solution(spd_matrix, rhs, rtol=rtol, margin=margin)
    relative_residual = float(
        torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(rhs)
    )
    assert relative_residual <= rtol * margin


def test_reference_precision_margin_clamps_to_machine_epsilon(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    """A margin*rtol below machine precision is clamped, not asked for literally."""
    rtol, margin = 1e-14, 1e-4  # requested target ~1e-18, below float64 precision
    x = reference_solution(spd_matrix, rhs, rtol=rtol, margin=margin)
    relative_residual = float(
        torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(rhs)
    )
    assert relative_residual <= MIN_SAFE_RELATIVE_RESIDUAL * 10


def test_reference_precision_margin_rejects_non_tightening_margin(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="reference_precision_margin"):
        reference_solution(spd_matrix, rhs, rtol=1e-6, margin=1.0)
