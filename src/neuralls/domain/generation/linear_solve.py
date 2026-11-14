"""Direct and CG linear-system solve dispatch, and solution-accuracy measurement."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

from .matrix_operator import MatrixOperator
from .notices import SolveNotConverged
from .strategy_configs import SolveConfig


@dataclass(frozen=True, slots=True)
class SolvedSystems:
    """Solutions of a stack of systems, with the rows whose iterative solve stopped early.

    Attributes:
        solutions: Solution vectors, shape (num_systems, n).
        unconverged: One notice per row the solver did not converge on; always empty for
            direct solves.
    """

    solutions: np.ndarray
    unconverged: tuple[SolveNotConverged, ...] = ()


type _SolveFn = Callable[[MatrixOperator, np.ndarray, SolveConfig], SolvedSystems]
"""Solver over a stack of RHS rows, shape (num_systems, n)."""


def _solve_direct_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, config: SolveConfig
) -> SolvedSystems:
    """Direct solve per RHS row, reusing the operator's cached factor."""
    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    for idx, rhs in enumerate(rhs_array):
        solutions[idx] = operator.solve_direct(rhs, assume_pos_def=config.assume_pos_def)
    return SolvedSystems(solutions)


def _solve_cg_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, config: SolveConfig
) -> SolvedSystems:
    """Iterative CG per RHS row; rows that stop early are reported, not dropped."""
    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    unconverged: list[SolveNotConverged] = []
    for idx, rhs in enumerate(rhs_array):
        solution, exit_code = scipy_cg(
            operator.matrix,
            rhs,
            rtol=config.rtol,
            atol=config.atol,
            maxiter=config.max_iters,
        )
        solutions[idx] = solution
        if exit_code != 0:
            residual = float(np.linalg.norm(operator.matvec(solution) - rhs))
            unconverged.append(SolveNotConverged(row=idx, exit_code=exit_code, residual=residual))
    return SolvedSystems(solutions, tuple(unconverged))


type _SolveMethod = Literal["direct", "cg"]
"""Closed set of linear-solve strategies accepted by `_solve_linear_systems`."""

_SOLVERS: Mapping[_SolveMethod, _SolveFn] = MappingProxyType(
    {
        "direct": _solve_direct_rows,
        "cg": _solve_cg_rows,
    }
)


def _solve_linear_systems(
    A: MatrixOperator, rhs_vectors: np.ndarray, config: SolveConfig
) -> SolvedSystems:
    """Solve linear systems Ax = b with the configured method.

    Direct solves use the operator's cached factorization; CG works on either format.

    Args:
        A: Operator wrapping the system matrix, shape (n, n)
        rhs_vectors: RHS vectors, shape (num_systems, n)
        config: Validated solve settings; ``config.method`` selects the solver.

    Returns:
        The solutions and the rows CG did not converge on.
    """
    rhs_array = np.asarray(rhs_vectors, dtype=np.float64)
    return _SOLVERS[config.method](A, rhs_array, config)


def _verify_solution_accuracy(
    A: np.ndarray,
    rhs_vectors: np.ndarray,
    solutions: np.ndarray,
) -> np.ndarray:
    """Relative residual ``||A x - b|| / ||b||`` of each (solution, RHS) pair.

    A zero RHS falls back to the absolute residual.

    Args:
        A: System matrix
        rhs_vectors: Right-hand side vectors
        solutions: Solution vectors

    Returns:
        Array of relative residuals, shape (count,)
    """
    count = rhs_vectors.shape[0]
    rel_residuals = np.zeros(count)
    for i in range(count):
        residual = A @ solutions[i] - rhs_vectors[i]
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_residuals[i] = (
            np.linalg.norm(residual) / rhs_norm if rhs_norm > 0 else np.linalg.norm(residual)
        )
    return rel_residuals
