"""Direct and CG linear-system solve dispatch, and solution-accuracy verification."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from loguru import logger

from .matrix_operator import MatrixOperator


@dataclass(frozen=True, slots=True)
class _SolveOptions:
    """Per-call solver settings shared by every registry entry."""

    rtol: float
    atol: float
    max_iters: int
    assume_pos_def: bool


type _SolveFn = Callable[[MatrixOperator, np.ndarray, _SolveOptions], np.ndarray]
"""Solver over a stack of RHS rows, shape (num_systems, n)."""


def _solve_direct_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, options: _SolveOptions
) -> np.ndarray:
    """Direct solve per RHS row, reusing the operator's cached factor."""
    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    for idx, rhs in enumerate(rhs_array):
        solutions[idx] = operator.solve_direct(rhs, assume_pos_def=options.assume_pos_def)
    return solutions


def _solve_cg_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, options: _SolveOptions
) -> np.ndarray:
    """Iterative CG per RHS row; warns on non-convergence."""
    from scipy.sparse.linalg import cg as scipy_cg

    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    for idx, rhs in enumerate(rhs_array):
        solution, exit_code = scipy_cg(
            operator.matrix,
            rhs,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iters,
        )
        solutions[idx] = solution

        if exit_code != 0:
            residual = np.linalg.norm(operator.matvec(solution) - rhs)
            logger.warning(f"System {idx + 1} CG exit_code={exit_code} (residual: {residual:.2e})")
    return solutions


type _SolveMethod = Literal["direct", "cg"]
"""Closed set of linear-solve strategies accepted by `_solve_linear_systems`."""

_SOLVERS: Mapping[_SolveMethod, _SolveFn] = MappingProxyType(
    {
        "direct": _solve_direct_rows,
        "cg": _solve_cg_rows,
    }
)


def _solve_linear_systems(
    A: MatrixOperator,
    rhs_vectors: np.ndarray,
    method: _SolveMethod,
    rtol: float = 1e-12,
    atol: float = 0.0,
    max_iters: int = 500,
    assume_pos_def: bool = True,
) -> np.ndarray:
    """Solve linear systems Ax = b using configured method and matrix format.

    The solver is looked up in `_SOLVERS` by method. Direct solves use the
    operator's cached factorization; CG works on either format.

    Args:
        A: Operator wrapping the system matrix, shape (n, n)
        rhs_vectors: RHS vectors, shape (num_systems, n)
        method: Solving method ("direct" or "cg")
        rtol: Relative tolerance for CG (ignored for direct)
        atol: Absolute tolerance for CG (ignored for direct)
        max_iters: Maximum CG iterations (ignored for direct)
        assume_pos_def: Dense direct solve only: Cholesky (True) or LU (False)

    Returns:
        Solution vectors, shape (num_systems, n)

    Raises:
        ValueError: If no solver is registered for `method`
    """
    rhs_array = np.asarray(rhs_vectors, dtype=np.float64)
    solver = _SOLVERS.get(method)
    if solver is None:
        raise ValueError(f"Invalid solve method: {method}. Must be 'direct' or 'cg'")
    options = _SolveOptions(
        rtol=rtol,
        atol=atol,
        max_iters=max_iters,
        assume_pos_def=assume_pos_def,
    )
    return solver(A, rhs_array, options)


def _verify_solution_accuracy(
    A: np.ndarray,
    rhs_vectors: np.ndarray,
    solutions: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Verify solution accuracy by computing relative residuals.

    Args:
        A: System matrix
        rhs_vectors: Right-hand side vectors
        solutions: Solution vectors
        tolerance: Relative residual tolerance for warnings

    Returns:
        Array of relative residuals, shape (count,)

    Warns:
        RuntimeWarning: If any solution exceeds tolerance
    """
    count = rhs_vectors.shape[0]
    rel_residuals = np.zeros(count)
    for i in range(count):
        residual = A @ solutions[i] - rhs_vectors[i]
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_residuals[i] = (
            np.linalg.norm(residual) / rhs_norm if rhs_norm > 0 else np.linalg.norm(residual)
        )
        if rel_residuals[i] > tolerance:
            warnings.warn(
                f"Sample {i}: Solution accuracy {rel_residuals[i]:.2e} "
                f"exceeds tolerance {tolerance:.2e}",
                RuntimeWarning,
                stacklevel=2,
            )
    return rel_residuals
