"""Eigendecomposition-backed sample generation: selection and combination of eigenvectors."""

from __future__ import annotations

import numpy as np

from neuralls.shared.types import EigenvectorSelection, MatrixFormat

from .matrix_operator import MatrixOperator, require_eigen_count


def _compute_eigendecomposition(
    operator: MatrixOperator,
    count: int,
    which: EigenvectorSelection,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenpairs needed to select `count` eigenvectors.

    "smallest" and "largest" ask the operator for only `count` pairs, which
    is the only option for CSR. "random" needs the whole spectrum, so it is
    dense-only; the dense full spectrum is requested as `count = n`.

    Args:
        operator: Operator wrapping the symmetric system matrix
        count: Number of eigenpairs to return for "smallest"/"largest"
        which: Selection mode

    Returns:
        Tuple of (eigenvalues ascending, eigenvectors as columns)

    Raises:
        ValueError: If the matrix is not symmetric, or "random" is requested
            for a CSR operator
    """
    match which:
        case EigenvectorSelection.RANDOM:
            if operator.format is MatrixFormat.CSR:
                raise ValueError(
                    "Random eigenvector selection needs the full spectrum, which requires "
                    "a dense matrix; use 'smallest' or 'largest' with csr matrices"
                )
            n = operator.shape[0]
            return operator.eigensystem(n, EigenvectorSelection.SMALLEST)
        case EigenvectorSelection.SMALLEST | EigenvectorSelection.LARGEST:
            return operator.eigensystem(count, which)


def _select_eigenvectors(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    count: int,
    which: EigenvectorSelection,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select subset of eigenvectors according to which eigenvalues to use.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, n)
        eigenvalues: Eigenvalue array, shape (n,)
        count: Number of eigenvectors to select
        which: Which eigenvalues to select ("smallest", "largest", or "random")
            - "smallest": Select k eigenvectors with smallest eigenvalues
            - "largest": Select k eigenvectors with largest eigenvalues
            - "random": Random selection without replacement
        rng: Random number generator (used for "random" mode)

    Returns:
        Tuple of (selected_eigenvectors, selected_eigenvalues, indices)

    Raises:
        ValueError: If count is outside 1..n.
    """
    n = eigenvectors.shape[0]
    available = eigenvalues.shape[0]
    require_eigen_count(count, n)
    match which:
        case EigenvectorSelection.SMALLEST:
            indices = np.arange(count)
        case EigenvectorSelection.LARGEST:
            indices = np.arange(available - count, available)
        case EigenvectorSelection.RANDOM:
            indices = rng.choice(available, size=count, replace=False)
    return eigenvectors[:, indices], eigenvalues[indices], indices


def _generate_eigenvector_combinations(
    eigenvectors: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random L2-normalized linear combinations of eigenvectors.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, k)
        num_samples: Number of combinations to generate
        rng: Random number generator

    Returns:
        Linear combinations, shape (num_samples, n)
    """
    _n, k = eigenvectors.shape
    coeffs = rng.standard_normal(size=(num_samples, k), dtype=np.float64)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True)
    coeffs_normalized = coeffs / norms
    return coeffs_normalized @ eigenvectors.T
