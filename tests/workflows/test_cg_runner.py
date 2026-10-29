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
from torchalg.models.result import SolverResult
from torchalg.preconditioners.base import NonLinearPreconditioner, PreconditionerContext
from torchalg.preconditioners.implementations import (
    Identity,
)

from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
    StandaloneMethodPresentation,
)
from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.domain.solver.comparison_summary import (
    format_results_summary,
    summarize_best_combinations,
)
from neuralls.domain.solver.error_metrics import relative_exact_error
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy

pytestmark = pytest.mark.deferred_io("config-loading")

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


def test_run_cg_comparison_dispatches_by_preconditioner_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    spd_matrix: Tensor,
    rhs_vector: Tensor,
) -> None:
    """Non-linear preconditioners route to flexible_cg; SPD ones route to pcg."""

    class DummyNonLinearPreconditioner(NonLinearPreconditioner):
        def setup(
            self,
            matrix: torch.Tensor,
            context: PreconditionerContext | None = None,
        ) -> DummyNonLinearPreconditioner:
            self._mark_ready()
            return self

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
            "nonlinear": DummyNonLinearPreconditioner().setup(spd_matrix),
            "identity": Identity(),
        },
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    assert calls == ["flexible", "pcg"]
    assert results["nonlinear"].converged
    assert results["identity"].converged


def test_relative_exact_error_preserves_dtype_and_computes_correctly(
    close_solution_pair: tuple[Tensor, Tensor],
) -> None:
    """relative_exact_error computes the correct ratio at both float32 and float64."""
    x_sol, x_exact = close_solution_pair

    error = relative_exact_error(x_sol, x_exact)

    expected = float(torch.linalg.vector_norm(x_sol - x_exact) / torch.linalg.vector_norm(x_exact))
    assert error == pytest.approx(expected, rel=1e-5)


def test_relative_exact_error_zero_exact_norm_returns_absolute_diff(
    zero_exact_solution_pair: tuple[Tensor, Tensor],
) -> None:
    """The guard clause returns the absolute diff norm when x_exact is the zero vector."""
    x_sol, x_exact = zero_exact_solution_pair

    error = relative_exact_error(x_sol, x_exact)

    assert error == pytest.approx(1.0)


# ==============================================================================
# Tests for format_results_summary()
# ==============================================================================


@pytest.fixture
def result_methods() -> dict[str, StandaloneMethodPresentation]:
    """The scientific description of each mock result's method."""
    return {
        "jacobi": StandaloneMethodPresentation(ScientificDescription("Jacobi")),
        "identity": StandaloneMethodPresentation(ScientificDescription("Identity")),
        "failed": StandaloneMethodPresentation(ScientificDescription("Broken preconditioner")),
    }


def test_format_results_summary(
    mock_comparison_results: dict[str, CGComparisonResult],
    result_methods: dict[str, StandaloneMethodPresentation],
) -> None:
    """Each summary line is headed by the method's description, never its id."""
    summary = format_results_summary(
        mock_comparison_results, result_methods, NotationPolicy.default()
    )

    assert summary == (
        "CG comparison results:\n"
        "- Jacobi             status=ok   iters= 10  rel_res=1.000e-08 (abs=1.000e-09)  exact_err=1.000e-10\n"
        "- Identity           status=ok   iters= 20  rel_res=1.000e-07 (abs=1.000e-08)  exact_err=5.000e-10\n"
        "- Broken preconditioner status=fail iters=100  rel_res=5.000e-01 (abs=5.000e-01)  note=breakdown detected  note=breakdown"
    )


def test_format_results_summary_rejects_a_result_without_a_description(
    mock_comparison_results: dict[str, CGComparisonResult],
    result_methods: dict[str, StandaloneMethodPresentation],
) -> None:
    del result_methods["identity"]

    with pytest.raises(MissingMethodPresentationError):
        format_results_summary(mock_comparison_results, result_methods, NotationPolicy.default())


def test_format_results_summary_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
    result_methods: dict[str, StandaloneMethodPresentation],
) -> None:
    """Test format_results_summary with converged results."""
    # Keep only converged results
    converged_results = {k: v for k, v in mock_comparison_results.items() if v.converged}

    summary = format_results_summary(converged_results, result_methods, NotationPolicy.default())

    # Should only have "ok" status
    assert "status=ok" in summary
    assert "status=fail" not in summary


def test_format_results_summary_non_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
    result_methods: dict[str, StandaloneMethodPresentation],
) -> None:
    """Test format_results_summary with non-converged results."""
    # Keep only non-converged results
    non_converged = {k: v for k, v in mock_comparison_results.items() if not v.converged}

    summary = format_results_summary(non_converged, result_methods, NotationPolicy.default())

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

    # Equal (zero) cost, so iterations break the tie: jacobi 10 < identity 20
    assert ranked[0].method_id == "jacobi"
    assert ranked[1].method_id == "identity"

    # Verify overall_best
    best = summary.overall_best
    assert best is not None
    assert best.method_id == "jacobi"
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
    assert summary.overall_best is not None

    # Verify single entry
    assert len(summary.ranked) == 1
    assert summary.overall_best.method_id == "jacobi"


def test_summarize_best_combinations_ranks_by_time_not_final_residual(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """The lowest residual is not the winner: the faster run wins despite the looser residual."""
    from dataclasses import replace

    slow_but_precise = replace(
        mock_comparison_results["jacobi"], solve_time_seconds=9.0, residual=1e-12
    )
    fast_but_looser = replace(
        mock_comparison_results["identity"], solve_time_seconds=1.0, residual=1e-7
    )

    summary = summarize_best_combinations({"jacobi": slow_but_precise, "identity": fast_but_looser})

    assert [entry.method_id for entry in summary.ranked] == ["identity", "jacobi"]
    assert summary.overall_best is not None
    assert summary.overall_best.solve_time_seconds == pytest.approx(1.0)
