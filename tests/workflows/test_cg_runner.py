"""Tests for CG comparison runner orchestration.

This module tests the workflow orchestration functions in cg_runner.py:
- run_cg_comparison(): Core CG comparison orchestration
- format_results_summary(): Result formatting
- summarize_best_combinations(): Analysis and ranking

Follows project testing principles:
- Use fixtures for all test data (no inline data creation)
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Mirror source structure (tests/workflows/ mirrors src/neuralls/workflows/)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torchalg.models.result import SolverResult
from torchalg.preconditioners.base import NonLinearPreconditioner, PreconditionerContext
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    ScheduledPreconditioner,
)

from neuralls.domain.solver.comparison import (
    _relative_exact_error,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from neuralls.domain.solver.models.result import CGComparisonResult

if TYPE_CHECKING:
    from torch import Tensor


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def spd_matrix() -> Tensor:
    """Small 10x10 SPD matrix for testing.

    Returns:
        Well-conditioned SPD matrix
    """
    n = 10
    A = np.random.RandomState(42).rand(n, n)
    spd = A.T @ A + np.eye(n) * 2.0
    return torch.as_tensor(spd, dtype=torch.float64)


@pytest.fixture
def rhs_vector(spd_matrix: Tensor) -> Tensor:
    """Compatible RHS vector for spd_matrix.

    Args:
        spd_matrix: Test matrix fixture

    Returns:
        RHS vector with same dimension as matrix
    """
    return torch.as_tensor(np.random.RandomState(42).rand(spd_matrix.shape[0]), dtype=torch.float64)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Dtype under test for input/output precision-fidelity coverage."""
    return request.param


@pytest.fixture
def spd_matrix_precision(precision_dtype: torch.dtype) -> Tensor:
    """Small SPD matrix built directly at the parametrized dtype.

    Args:
        precision_dtype: Dtype to build the matrix at.

    Returns:
        Well-conditioned SPD matrix at ``precision_dtype``.
    """
    n = 10
    a = np.random.RandomState(42).rand(n, n)
    spd = a.T @ a + np.eye(n) * 2.0
    return torch.as_tensor(spd, dtype=precision_dtype)


@pytest.fixture
def rhs_vector_precision(spd_matrix_precision: Tensor, precision_dtype: torch.dtype) -> Tensor:
    """RHS vector matching ``spd_matrix_precision``'s shape and dtype.

    Args:
        spd_matrix_precision: Matrix fixture to match the shape of.
        precision_dtype: Dtype to build the vector at.

    Returns:
        RHS vector at ``precision_dtype``.
    """
    return torch.as_tensor(
        np.random.RandomState(42).rand(spd_matrix_precision.shape[0]), dtype=precision_dtype
    )


@pytest.fixture
def precision_tolerances(precision_dtype: torch.dtype) -> tuple[float, float]:
    """(rtol, atol) sized to each dtype's headroom - loose for float32, tight for float64.

    Args:
        precision_dtype: Dtype under test.

    Returns:
        Tuple of (rtol, atol).
    """
    if precision_dtype == torch.float32:
        return 1e-4, 1e-6
    return 1e-8, 1e-10


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def close_solution_pair(request: pytest.FixtureRequest) -> tuple[Tensor, Tensor]:
    """A solved-vs-exact pair with a known, nonzero relative error, at a given dtype.

    Returns:
        Tuple of (x_sol, x_exact) tensors at the parametrized dtype.
    """
    dtype = request.param
    x_sol = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    x_exact = torch.tensor([1.0, 2.0, 4.0], dtype=dtype)
    return x_sol, x_exact


@pytest.fixture
def zero_exact_solution_pair() -> tuple[Tensor, Tensor]:
    """A solved-vs-exact pair where the exact solution is the zero vector.

    Returns:
        Tuple of (x_sol, x_exact) exercising the zero-norm guard clause.
    """
    return torch.tensor([1.0, 0.0]), torch.zeros(2)


@pytest.fixture
def mock_comparison_results() -> dict[str, CGComparisonResult]:
    """Mock comparison results for testing formatters.

    Returns:
        Dict mapping preconditioner names to mock results
    """
    x_sol = np.array([1.0, 2.0, 3.0])
    x0 = np.zeros(3)

    return {
        "jacobi": CGComparisonResult(
            x=x_sol,
            converged=True,
            iterations=10,
            residual=1e-8,
            residual_abs=1e-9,
            residual_history_rel=[1.0, 0.1, 1e-8],
            residual_history_abs=[1.0, 0.1, 1e-9],
            preconditioner="jacobi",
            initial_guess=x0,
            exact_error=1e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
        "identity": CGComparisonResult(
            x=x_sol,
            converged=True,
            iterations=20,
            residual=1e-7,
            residual_abs=1e-8,
            residual_history_rel=[1.0, 0.5, 1e-7],
            residual_history_abs=[1.0, 0.5, 1e-8],
            preconditioner="identity",
            initial_guess=x0,
            exact_error=5e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
        "failed": CGComparisonResult(
            x=x0,
            converged=False,
            iterations=100,
            residual=0.5,
            residual_abs=0.5,
            residual_history_rel=[1.0, 0.9, 0.5],
            residual_history_abs=[1.0, 0.9, 0.5],
            preconditioner="failed",
            initial_guess=x0,
            exact_error=None,
            rhs_norm=1.0,
            breakdown=True,
            error="breakdown detected",
        ),
    }


# ==============================================================================
# Tests for run_cg_comparison()
# ==============================================================================


def test_run_cg_comparison_with_preconditioner_instances(
    spd_matrix: Tensor, rhs_vector: Tensor
) -> None:
    """Test run_cg_comparison with Identity and Jacobi preconditioner instances."""
    preconditioners = {
        "identity": Identity(),
        "jacobi": JacobiPreconditioner(spd_matrix),
    }

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify all keys present (including auto-added "none")
    assert "identity" in results
    assert "jacobi" in results
    assert "none" in results  # Auto-added baseline

    # Verify result types
    for name, result in results.items():
        assert isinstance(result, CGComparisonResult)
        assert result.preconditioner == name
        assert result.iterations >= 0
        assert isinstance(result.converged, bool)


def test_run_cg_comparison_routes_to_flexible_cg(spd_matrix: Tensor, rhs_vector: Tensor) -> None:
    """Test that ScheduledPreconditioner routes to flexible_cg."""
    primary = JacobiPreconditioner(spd_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    preconditioners = {"scheduled": scheduled}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify flexible_cg was used (check for iteration history)
    result = results["scheduled"]
    assert isinstance(result, CGComparisonResult)
    # Flexible CG provides iteration history
    assert len(result.residual_history_rel) > 1


def test_run_cg_comparison_dispatches_by_preconditioner_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    spd_matrix: Tensor,
    rhs_vector: Tensor,
) -> None:
    """Non-linear preconditioners route to flexible_cg; SPD ones route to pcg."""

    class DummyNonLinearPreconditioner(NonLinearPreconditioner):
        def apply(
            self,
            residual: torch.Tensor,
            context: PreconditionerContext | None = None,
        ) -> torch.Tensor:
            return residual

    calls: list[str] = []

    def make_fake_solver(
        label: str,
    ) -> Callable[..., tuple[torch.Tensor, SolverResult]]:
        def fake_solver(*args: object, **kwargs: object) -> tuple[torch.Tensor, SolverResult]:
            calls.append(label)
            return torch.zeros_like(rhs_vector), SolverResult(
                converged=True,
                iterations=3,
                residual=1e-8,
                residual_abs=1e-8,
                rhs_norm=float(torch.linalg.vector_norm(rhs_vector)),
                breakdown=False,
                residual_history_rel=(1.0, 1e-8),
                residual_history_abs=(1.0, 1e-8),
                tol=1e-8,
                atol=1e-10,
            )

        return fake_solver

    monkeypatch.setattr(
        "neuralls.domain.solver.comparison.flexible_cg", make_fake_solver("flexible")
    )
    monkeypatch.setattr("neuralls.domain.solver.comparison.pcg", make_fake_solver("pcg"))

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners={
            "nonlinear": DummyNonLinearPreconditioner(),
            "none": Identity(),
        },
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    assert calls == ["flexible", "pcg"]
    assert results["nonlinear"].converged
    assert results["none"].converged


def test_run_cg_comparison_routes_to_pcg(spd_matrix: Tensor, rhs_vector: Tensor) -> None:
    """Identity preconditioner converges via the real, non-mocked pcg() path."""
    preconditioners = {"identity": Identity()}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    result = results["identity"]
    assert isinstance(result, CGComparisonResult)
    assert result.converged


def test_run_cg_comparison_adds_none_baseline(spd_matrix: Tensor, rhs_vector: Tensor) -> None:
    """Test that 'none' baseline is automatically added if missing."""
    preconditioners = {"jacobi": JacobiPreconditioner(spd_matrix)}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify "none" was added
    assert "none" in results
    assert "jacobi" in results

    # Verify "none" result is valid
    none_result = results["none"]
    assert isinstance(none_result, CGComparisonResult)
    assert none_result.preconditioner == "none"


def test_run_cg_comparison_computes_exact_error(spd_matrix: Tensor, rhs_vector: Tensor) -> None:
    """Test that run_cg_comparison computes exact error correctly."""
    preconditioners = {"identity": Identity()}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    result = results["identity"]

    # Verify exact error computed
    assert result.exact_error is not None
    assert result.exact_error >= 0

    # Compute expected exact error
    x_exact = np.linalg.solve(spd_matrix.numpy(), rhs_vector.numpy())
    exact_norm = np.linalg.norm(x_exact)
    expected_error = np.linalg.norm(result.x - x_exact) / exact_norm

    assert_allclose(result.exact_error, expected_error, rtol=1e-7)


def test_run_cg_comparison_preserves_input_dtype(
    spd_matrix_precision: Tensor,
    rhs_vector_precision: Tensor,
    precision_dtype: torch.dtype,
    precision_tolerances: tuple[float, float],
) -> None:
    """CGComparisonResult.x keeps the input dtype through the full comparison pipeline.

    Device auto-placement in run_cg_comparison must never coerce precision -
    float32 in stays float32 out, float64 in stays float64 out.
    """
    rtol, atol = precision_tolerances

    results = run_cg_comparison(
        spd_matrix_precision,
        rhs_vector_precision,
        preconditioners={"identity": Identity()},
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    result = results["identity"]
    expected_numpy_dtype = torch.empty(0, dtype=precision_dtype).numpy().dtype
    assert result.x.dtype == expected_numpy_dtype
    assert result.exact_error is not None
    assert np.isfinite(result.exact_error)


def test_relative_exact_error_preserves_dtype_and_computes_correctly(
    close_solution_pair: tuple[Tensor, Tensor],
) -> None:
    """_relative_exact_error computes the correct ratio at both float32 and float64."""
    x_sol, x_exact = close_solution_pair

    error = _relative_exact_error(x_sol, x_exact)

    expected = float(torch.linalg.vector_norm(x_sol - x_exact) / torch.linalg.vector_norm(x_exact))
    assert error == pytest.approx(expected, rel=1e-5)


def test_relative_exact_error_zero_exact_norm_returns_absolute_diff(
    zero_exact_solution_pair: tuple[Tensor, Tensor],
) -> None:
    """The guard clause returns the absolute diff norm when x_exact is the zero vector."""
    x_sol, x_exact = zero_exact_solution_pair

    error = _relative_exact_error(x_sol, x_exact)

    assert error == pytest.approx(1.0)


# ==============================================================================
# Tests for format_results_summary()
# ==============================================================================


def test_format_results_summary(mock_comparison_results: dict[str, CGComparisonResult]) -> None:
    """Test that format_results_summary produces correct output format."""
    summary = format_results_summary(mock_comparison_results)

    # Verify header present
    assert "CG comparison results:" in summary

    # Verify all preconditioners listed
    assert "jacobi" in summary
    assert "identity" in summary
    assert "failed" in summary

    # Verify status codes
    assert "status=ok" in summary  # For converged results
    assert "status=fail" in summary  # For failed result

    # Verify breakdown note
    assert "breakdown" in summary


def test_format_results_summary_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test format_results_summary with converged results."""
    # Keep only converged results
    converged_results = {k: v for k, v in mock_comparison_results.items() if v.converged}

    summary = format_results_summary(converged_results)

    # Should only have "ok" status
    assert "status=ok" in summary
    assert "status=fail" not in summary


def test_format_results_summary_non_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test format_results_summary with non-converged results."""
    # Keep only non-converged results
    non_converged = {k: v for k, v in mock_comparison_results.items() if not v.converged}

    summary = format_results_summary(non_converged)

    # Should only have "fail" status
    assert "status=fail" in summary
    assert "status=ok" not in summary


# ==============================================================================
# Tests for summarize_best_combinations()
# ==============================================================================


def test_summarize_best_combinations(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test summarize_best_combinations ranks results correctly."""
    summary = summarize_best_combinations(mock_comparison_results)

    # Verify ranking (only converged results)
    ranked = summary.ranked
    assert len(ranked) == 2  # Only jacobi and identity converged

    # Verify sorted by residual (jacobi has lower residual: 1e-8 < 1e-7)
    assert ranked[0].label == "jacobi"
    assert ranked[1].label == "identity"

    # Verify overall_best
    best = summary.overall_best
    assert best is not None
    assert best.label == "jacobi"
    assert best.residual == 1e-8


def test_summarize_best_combinations_empty_results() -> None:
    """Test summarize_best_combinations with no converged results."""
    # Create all non-converged results
    non_converged_results = {
        "failed1": CGComparisonResult(
            x=np.zeros(3),
            converged=False,
            iterations=100,
            residual=0.5,
            residual_abs=0.5,
            residual_history_rel=[],
            residual_history_abs=[],
            preconditioner="failed1",
            initial_guess=np.zeros(3),
            exact_error=None,
            rhs_norm=1.0,
            breakdown=True,
        ),
    }

    summary = summarize_best_combinations(non_converged_results)

    # Verify empty ranking
    assert summary.ranked == ()
    assert summary.overall_best is None


def test_summarize_best_combinations_single_converged() -> None:
    """Test summarize_best_combinations with single converged result."""
    single_result = {
        "jacobi": CGComparisonResult(
            x=np.array([1.0, 2.0]),
            converged=True,
            iterations=10,
            residual=1e-8,
            residual_abs=1e-9,
            residual_history_rel=[],
            residual_history_abs=[],
            preconditioner="jacobi",
            initial_guess=np.zeros(2),
            exact_error=1e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
    }

    summary = summarize_best_combinations(single_result)

    # Verify single entry
    assert len(summary.ranked) == 1
    assert summary.overall_best is not None
    assert summary.overall_best.label == "jacobi"
