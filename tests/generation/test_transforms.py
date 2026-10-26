"""Tests for transformation layer (Layer 2: Pure Computation)."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.transforms import (
    ComputeRhsTransform,
    EigenvectorCombinationTransform,
    SolveTransform,
    VerifyTransform,
)


@pytest.fixture
def sample_spd_matrix() -> np.ndarray:
    """Sample 10x10 symmetric positive-definite matrix."""
    A = np.random.randn(10, 10)
    return A.T @ A + np.eye(10) * 0.1


@pytest.fixture
def diagonal_matrix() -> np.ndarray:
    """Simple diagonal matrix for testing."""
    return np.eye(10) * 2.0


def test_compute_rhs_transform_forward(diagonal_matrix: np.ndarray) -> None:
    """Test forward transform: b = A @ x."""
    transform = ComputeRhsTransform(diagonal_matrix)
    x = np.ones((5, 10))

    b = transform.transform(x)

    assert b.shape == (5, 10)
    assert b.dtype == np.float64
    # For diagonal matrix: b = 2 * x
    assert np.allclose(b, 2.0 * x)


def test_solve_transform_inverse_direct(diagonal_matrix: np.ndarray) -> None:
    """Test inverse transform with direct solve."""
    transform = SolveTransform(diagonal_matrix, method="direct")
    b = np.ones((5, 10)) * 2.0

    x = transform.transform(b)

    assert x.shape == (5, 10)
    assert x.dtype == np.float64
    # For diagonal matrix: x = b / 2
    assert np.allclose(x, np.ones((5, 10)))


def test_solve_transform_inverse_cg(sample_spd_matrix: np.ndarray) -> None:
    """Test inverse transform with CG solve."""
    transform = SolveTransform(
        sample_spd_matrix,
        method="cg",
        rtol=1e-12,
        max_iters=500,
    )

    # Create RHS from known solution
    x_true = np.random.randn(5, 10)
    b = np.array([sample_spd_matrix @ x for x in x_true])

    x_solved = transform.transform(b)

    assert x_solved.shape == (5, 10)
    # Check that solved x is close to true x
    for i in range(5):
        residual = sample_spd_matrix @ x_solved[i] - b[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(b[i])
        assert rel_residual < 1e-10


def test_solve_transform_round_trip(sample_spd_matrix: np.ndarray) -> None:
    """Test round-trip: x -> b -> x."""
    forward = ComputeRhsTransform(sample_spd_matrix)
    inverse = SolveTransform(sample_spd_matrix, method="cg", rtol=1e-12)

    x_original = np.random.randn(5, 10)
    b = forward.transform(x_original)
    x_recovered = inverse.transform(b)

    # Check round-trip accuracy
    assert np.allclose(x_original, x_recovered, rtol=1e-8, atol=1e-10)


def test_verify_transform_accurate_solutions(sample_spd_matrix: np.ndarray) -> None:
    """Test verification transform with accurate solutions."""
    transform = VerifyTransform(sample_spd_matrix, tolerance=1e-10)

    x = np.random.randn(5, 10)
    b = np.array([sample_spd_matrix @ xi for xi in x])

    residuals = transform.transform((x, b))

    assert residuals.shape == (5,)
    assert np.all(residuals < 1e-10)


def test_verify_transform_inaccurate_solutions(sample_spd_matrix: np.ndarray) -> None:
    """Test verification transform detects inaccurate solutions."""
    transform = VerifyTransform(sample_spd_matrix, tolerance=1e-10)

    x = np.random.randn(5, 10)
    b_wrong = np.random.randn(5, 10)  # Random RHS, not A @ x

    # Should warn about inaccurate solutions
    with pytest.warns(RuntimeWarning):
        residuals = transform.transform((x, b_wrong))

    assert residuals.shape == (5,)
    # Residuals should be large
    assert np.all(residuals > 0.1)


def test_eigenvector_combination_transform(sample_spd_matrix: np.ndarray) -> None:
    """Test eigenvector combination transform."""
    rng = np.random.default_rng(42)
    transform = EigenvectorCombinationTransform(
        count=5,
        which="smallest",
        rng=rng,
    )

    x = transform.transform(sample_spd_matrix)

    assert x.shape == (5, 10)
    assert x.dtype == np.float64
    # Check that vectors are L2-normalized
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_eigenvector_combination_transform_largest(
    sample_spd_matrix: np.ndarray,
) -> None:
    """Test eigenvector combination with largest eigenvalues."""
    rng = np.random.default_rng(42)
    transform = EigenvectorCombinationTransform(
        count=3,
        which="largest",
        rng=rng,
    )

    x = transform.transform(sample_spd_matrix)

    assert x.shape == (3, 10)
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_eigenvector_combination_transform_random(
    sample_spd_matrix: np.ndarray,
) -> None:
    """Test eigenvector combination with random selection."""
    rng = np.random.default_rng(42)
    transform = EigenvectorCombinationTransform(
        count=7,
        which="random",
        rng=rng,
    )

    x = transform.transform(sample_spd_matrix)

    assert x.shape == (7, 10)
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-10)


def test_transforms_are_composable(sample_spd_matrix: np.ndarray) -> None:
    """Test that transforms can be chained."""
    rng = np.random.default_rng(42)

    # Build pipeline: Eigenvector combinations -> Compute RHS -> Solve -> Verify
    eigen_transform = EigenvectorCombinationTransform(
        count=5,
        which="smallest",
        rng=rng,
    )
    forward_transform = ComputeRhsTransform(sample_spd_matrix)
    inverse_transform = SolveTransform(sample_spd_matrix, method="cg", rtol=1e-12)
    verify_transform = VerifyTransform(sample_spd_matrix, tolerance=1e-10)

    # Execute pipeline
    x_original = eigen_transform.transform(sample_spd_matrix)
    b = forward_transform.transform(x_original)
    x_recovered = inverse_transform.transform(b)
    residuals = verify_transform.transform((x_recovered, b))

    # Verify pipeline correctness
    assert np.all(residuals < 1e-10)
    assert np.allclose(x_original, x_recovered, rtol=1e-8, atol=1e-10)
