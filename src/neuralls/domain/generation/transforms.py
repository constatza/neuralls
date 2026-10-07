"""Transformation layer for generation strategies (SOLID Layer 2: Pure Computation).

Transforms encapsulate pure computational operations on input data, independent of
data sources. This separation enables composition, reusability, and testing.

Architecture:
    - Transformation: Protocol for any pure computation
    - Concrete implementations: ComputeRhs, Solve, Verify, CGTrace, etc.
    - Strategies compose transforms to build data pipelines

Examples:
    >>> # Forward: b = A @ x
    >>> transform = ComputeRhsTransform(matrix)
    >>> rhs = transform.transform(solutions)

    >>> # Inverse: x = A^-1 @ b
    >>> transform = SolveTransform(matrix, method="cg")
    >>> solutions = transform.transform(rhs)

    >>> # Chaining transforms
    >>> eigen_transform = EigenvectorCombinationTransform(config, rng)
    >>> rhs_transform = ComputeRhsTransform(matrix)
    >>> solutions = eigen_transform.transform(matrix)
    >>> rhs = rhs_transform.transform(solutions)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, Protocol, TypeVar

import numpy as np
from scipy.sparse.linalg import LinearOperator

from neuralls.shared.types import EigenvectorSelection, SystemMatrix

from .eigen_strategies import (
    _compute_eigendecomposition,
    _generate_eigenvector_combinations,
    _select_eigenvectors,
)
from .krylov import _generate_krylov_combinations, _lanczos_iteration
from .linear_solve import (
    _solve_linear_systems,
    _verify_solution_accuracy,
)
from .matrix_operator import MatrixOperator, require_eigen_count

TIn_contra = TypeVar("TIn_contra", contravariant=True)
TOut_co = TypeVar("TOut_co", covariant=True)


class Transformation(Protocol[TIn_contra, TOut_co]):
    """Protocol for pure computational transformations.

    Transformations are deterministic functions that convert inputs to outputs
    without side effects. They should be composable and testable in isolation.
    """

    @abstractmethod
    def transform(self, inputs: TIn_contra) -> TOut_co:
        """Apply transformation to inputs.

        Args:
            inputs: Input data of type TIn_contra

        Returns:
            Transformed data of type TOut_co
        """
        ...


class ComputeRhsTransform:
    """Forward transformation: b = A @ x.

    Examples:
        >>> A = np.eye(10) * 2
        >>> transform = ComputeRhsTransform(A)
        >>> x = np.ones((5, 10))
        >>> b = transform.transform(x)
        >>> b.shape
        (5, 10)
        >>> np.allclose(b, 2 * x)
        True
    """

    def __init__(self, matrix: np.ndarray | LinearOperator | Any) -> None:
        """Initialize forward transform.

        Args:
            matrix: System matrix A
        """
        self.matrix = matrix

    def transform(self, solutions: np.ndarray) -> np.ndarray:
        """Compute RHS: b = A @ x.

        Args:
            solutions: Solution vectors, shape (N, n)

        Returns:
            RHS vectors, shape (N, n)
        """
        return np.asarray([self.matrix @ x for x in solutions], dtype=np.float64)


class SolveTransform:
    """Inverse transformation: x = A^-1 @ b.

    Examples:
        >>> A = np.eye(10) * 2
        >>> transform = SolveTransform(A, method="direct")
        >>> b = np.ones((5, 10)) * 2
        >>> x = transform.transform(b)
        >>> x.shape
        (5, 10)
        >>> np.allclose(x, np.ones((5, 10)))
        True
    """

    def __init__(
        self,
        matrix: SystemMatrix,
        method: Literal["direct", "cg"] = "cg",
        rtol: float = 1e-12,
        atol: float = 0.0,
        max_iters: int = 500,
        assume_pos_def: bool = True,
    ) -> None:
        """Initialize solve transform.

        Args:
            matrix: System matrix A
            method: Solving method ("direct" or "cg")
            rtol: Relative tolerance for CG
            atol: Absolute tolerance for CG
            max_iters: Maximum CG iterations
            assume_pos_def: Assume positive-definite for direct solve
        """
        self.operator = MatrixOperator(matrix)
        self.method: Literal["direct", "cg"] = method
        self.rtol = rtol
        self.atol = atol
        self.max_iters = max_iters
        self.assume_pos_def = assume_pos_def

    def transform(self, rhs: np.ndarray) -> np.ndarray:
        """Solve systems: x = A^-1 @ b.

        Args:
            rhs: RHS vectors, shape (N, n)

        Returns:
            Solution vectors, shape (N, n)
        """
        return _solve_linear_systems(
            self.operator,
            rhs,
            self.method,
            self.rtol,
            self.atol,
            self.max_iters,
            self.assume_pos_def,
        )


class VerifyTransform:
    """Verification transformation: check ||A @ x - b|| < tolerance.

    Examples:
        >>> A = np.eye(10)
        >>> transform = VerifyTransform(A, tolerance=1e-10)
        >>> x = np.ones((5, 10))
        >>> b = A @ x[0]  # Use first solution
        >>> b_full = np.array([b] * 5)
        >>> residuals = transform.transform((x, b_full))
        >>> residuals.shape
        (5,)
        >>> np.all(residuals < 1e-10)
        True
    """

    def __init__(self, matrix: np.ndarray, tolerance: float = 1e-10) -> None:
        """Initialize verification transform.

        Args:
            matrix: System matrix A
            tolerance: Relative residual tolerance
        """
        self.matrix = matrix
        self.tolerance = tolerance

    def transform(self, data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Verify solution accuracy.

        Args:
            data: Tuple of (solutions, rhs), both shape (N, n)

        Returns:
            Relative residuals, shape (N,)
        """
        solutions, rhs = data
        return _verify_solution_accuracy(
            self.matrix,
            rhs,
            solutions,
            self.tolerance,
        )


class EigenvectorCombinationTransform:
    """Eigenvector-based transformation: generate linear combinations.

    This transform computes eigendecomposition, selects eigenvectors according
    to the specified strategy, and generates random L2-normalized linear combinations.

    Features:
        - Select eigenvectors by eigenvalue (smallest, largest, random)
        - Optionally include pure eigenvectors in output
        - Control number of eigenvectors vs number of combinations

    Examples:
        >>> A = create_spd_matrix(10)
        >>> # Generate 5 random combinations from smallest eigenvectors
        >>> transform = EigenvectorCombinationTransform(
        ...     count=5, which="smallest", rng=np.random.default_rng(42)
        ... )
        >>> solutions = transform.transform(A)
        >>> solutions.shape
        (5, 10)

        >>> # Include 3 pure eigenvectors + 7 combinations (total 10)
        >>> transform = EigenvectorCombinationTransform(
        ...     count=10,
        ...     which="smallest",
        ...     rng=np.random.default_rng(42),
        ...     num_eigenvectors=3,
        ...     include_eigenvectors=True,
        ... )
        >>> solutions = transform.transform(A)
        >>> solutions.shape
        (10, 10)
    """

    def __init__(
        self,
        count: int,
        which: EigenvectorSelection,
        rng: np.random.Generator,
        num_eigenvectors: int | None = None,
        include_eigenvectors: bool = False,
    ) -> None:
        """Initialize eigenvector combination transform.

        Args:
            count: Total number of vectors to generate
            which: Which eigenvectors ("smallest", "largest", "random")
            rng: Random number generator
            num_eigenvectors: Number of eigenvectors to use for combinations
                (defaults to count). Can be less than count when include_eigenvectors=True
            include_eigenvectors: Whether to include pure eigenvectors in output
        """
        self.count = count
        self.which = which
        self.rng = rng
        self.num_eigenvectors = num_eigenvectors
        self.include_eigenvectors = include_eigenvectors

    def transform(self, matrix: SystemMatrix) -> np.ndarray:
        """Generate eigenvector combinations.

        Args:
            matrix: System matrix A (dense or CSR)

        Returns:
            Vectors (eigenvectors and/or combinations), shape (count, n)
        """
        operator = MatrixOperator(matrix)
        n = operator.shape[0]

        # Resolve num_eigenvectors
        if self.num_eigenvectors is None:
            num_eigvecs = self.count
        else:
            num_eigvecs = self.num_eigenvectors
            if num_eigvecs == -1:
                num_eigvecs = n

        require_eigen_count(num_eigvecs, n)

        # Only the selected eigenpairs are computed; CSR cannot provide the full spectrum
        eigenvalues, eigenvectors = _compute_eigendecomposition(operator, num_eigvecs, self.which)

        # Select subset of eigenvectors
        selected_eigvecs, _, _ = _select_eigenvectors(
            eigenvectors,
            eigenvalues,
            num_eigvecs,
            self.which,
            self.rng,
        )

        # Generate combinations (with optional inclusion of pure eigenvectors)
        if self.include_eigenvectors:
            num_combinations = self.count - num_eigvecs
            if num_combinations < 0:
                raise ValueError(
                    f"When include_eigenvectors=True, count ({self.count}) must be >= "
                    f"num_eigenvectors ({num_eigvecs})."
                )
            if num_combinations > 0:
                combinations = _generate_eigenvector_combinations(
                    selected_eigvecs, num_combinations, self.rng
                )
                # Stack eigenvectors (as rows) with combinations
                return np.vstack([selected_eigvecs.T, combinations])
            else:
                # Only eigenvectors, no combinations
                return selected_eigvecs.T
        else:
            # Only combinations, no pure eigenvectors
            return _generate_eigenvector_combinations(selected_eigvecs, self.count, self.rng)


class KrylovBasisTransform:
    """Krylov subspace transformation via Lanczos iteration.

    This transform builds a Krylov subspace basis using Lanczos iteration and
    generates linear combinations from the basis vectors.

    The Lanczos algorithm builds an orthonormal basis for the Krylov subspace
    K_m(A, v) = span{v, Av, A²v, ..., A^(m-1)v} and simultaneously constructs
    a tridiagonal matrix T that represents the projection of A onto this subspace.

    Examples:
        >>> A = create_spd_matrix(10)
        >>> transform = KrylovBasisTransform(
        ...     krylov_dim=5, num_samples=10, rng=np.random.default_rng(42)
        ... )
        >>> solutions = transform.transform(A)
        >>> solutions.shape
        (10, 10)
    """

    def __init__(
        self,
        krylov_dim: int,
        num_samples: int,
        rng: np.random.Generator,
    ) -> None:
        """Initialize Krylov basis transform.

        Args:
            krylov_dim: Dimension of Krylov subspace (number of Lanczos iterations)
            num_samples: Number of linear combinations to generate
            rng: Random number generator
        """
        self.krylov_dim = krylov_dim
        self.num_samples = num_samples
        self.rng = rng

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Build Krylov basis and generate combinations.

        Args:
            matrix: System matrix A

        Returns:
            Linear combinations of Krylov basis, shape (num_samples, n)
        """
        # Build Krylov subspace via Lanczos iteration
        V, T = _lanczos_iteration(matrix, self.krylov_dim, self.rng)

        # Generate random combinations from the basis
        return _generate_krylov_combinations(V, T, self.num_samples, self.rng)


class CGTraceTransform:
    """CG tracing transformation: run CG and collect iteration data.

    This transform runs CG solver on provided RHS vectors and collects iteration
    traces (residuals, errors, search directions) depending on the mode.

    Note: Full implementation requires CG iteration history tracking.
    This is a placeholder for Phase 5 implementation.

    Examples:
        >>> A = create_spd_matrix(10)
        >>> transform = CGTraceTransform(A, mode="residuals", max_iters=50)
        >>> b = np.random.randn(5, 10)
        >>> traces = transform.transform((b, None))
        >>> len(traces)  # One trace per RHS
        5
    """

    def __init__(
        self,
        matrix: np.ndarray,
        mode: Literal["residuals", "errors", "directions"] = "residuals",
        max_iters: int = 50,
        rtol: float = 1e-12,
        atol: float = 0.0,
    ) -> None:
        """Initialize CG tracing transform.

        Args:
            matrix: System matrix A
            mode: What to collect ("residuals", "errors", "directions")
            max_iters: Maximum CG iterations
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.matrix = matrix
        self.mode = mode
        self.max_iters = max_iters
        self.rtol = rtol
        self.atol = atol

    def transform(self, data: tuple[np.ndarray, np.ndarray | None]) -> list[np.ndarray]:
        """Run CG and collect traces.

        Args:
            data: Tuple of (rhs, true_solutions)
                - rhs: RHS vectors, shape (N, n)
                - true_solutions: Optional true solutions for error mode

        Returns:
            List of traces, one per RHS vector

        Note:
            This is a placeholder. Full implementation in Phase 5 will integrate
            with CG solver iteration history tracking.
        """
        raise NotImplementedError("CGTraceTransform - Phase 5")


__all__ = [
    "CGTraceTransform",
    "ComputeRhsTransform",
    "EigenvectorCombinationTransform",
    "KrylovBasisTransform",
    "SolveTransform",
    "Transformation",
    "VerifyTransform",
]
