"""Krylov-subspace (Lanczos) sample generation helpers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import norm


def _lanczos_iteration(
    matrix: np.ndarray,
    krylov_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Lanczos iteration to build Krylov subspace basis.

    Args:
        matrix: System matrix A
        krylov_dim: Dimension of Krylov subspace
        rng: Random number generator

    Returns:
        Tuple of (V, T) where:
            - V: Orthonormal basis vectors, shape (n, m_eff)
            - T: Tridiagonal matrix, shape (m_eff, m_eff)
            m_eff <= krylov_dim (early termination if breakdown occurs)
    """
    n = matrix.shape[0]
    m = krylov_dim
    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m + 1, dtype=np.float64)

    # Initial random vector
    v = rng.normal(size=n).astype(np.float64, copy=False)
    v = v / norm(v)
    V[:, 0] = v

    v_prev = np.zeros(n, dtype=np.float64)
    beta[0] = 0.0

    # Lanczos iteration
    m_eff = m
    for j in range(m):
        w = matrix @ V[:, j] - beta[j] * v_prev
        alpha[j] = np.dot(V[:, j], w)
        w = w - alpha[j] * V[:, j]
        beta[j + 1] = norm(w)

        # Check for breakdown (lucky breakdown)
        if beta[j + 1] <= 1e-14:
            m_eff = j + 1
            V = V[:, :m_eff]
            alpha = alpha[:m_eff]
            beta = beta[: m_eff + 1]
            break

        v_prev = V[:, j].copy()
        if j + 1 < m:
            V[:, j + 1] = w / beta[j + 1]

    # Build tridiagonal matrix
    T = np.diag(alpha[:m_eff]) + np.diag(beta[1:m_eff], k=-1) + np.diag(beta[1:m_eff], k=1)

    return V, T


def _generate_krylov_combinations(
    V: np.ndarray,
    T: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random linear combinations from Krylov basis.

    Args:
        V: Orthonormal Krylov basis, shape (n, m_eff)
        T: Tridiagonal matrix from Lanczos, shape (m_eff, m_eff)
        num_samples: Number of combinations to generate
        rng: Random number generator

    Returns:
        Linear combinations, shape (num_samples, n)
    """
    m_eff = V.shape[1]

    # Eigendecomposition of tridiagonal matrix
    Lambda, Q = np.linalg.eigh(T)
    Lambda_inv = 1.0 / Lambda

    # Generate random combinations
    combinations = []
    for _ in range(num_samples):
        eps = rng.normal(size=m_eff).astype(np.float64, copy=False)
        x = V @ (Q @ (Lambda_inv * eps))
        combinations.append(x)

    return np.array(combinations, dtype=np.float64)
