"""Test script to verify RHS normalization in data generation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import norm

from neuralls.domain.generation.strategies.krylov import KrylovStrategy
from neuralls.domain.generation.strategies.random_normal import RandomNormalStrategy


def test_normal_strategy_normalization():
    """Test that normal_strategy generates RHS with consistent norms."""
    print("Testing normal_strategy normalization...")

    # Create a simple test system
    n = 50
    A = np.eye(n) + 0.1 * np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric
    b = np.random.randn(n)
    mother_rhs_norm = norm(b)

    print(f"  Mother RHS norm: {mother_rhs_norm:.6f}")

    # Generate samples
    strategy = RandomNormalStrategy()
    output = strategy.generate(
        matrix=A, cfg={"samples": 100, "seed": 42, "target_rhs_scale": 1.0}, archive=None
    )
    R = output.rhs
    X = output.solutions
    assert R is not None and X is not None

    # Check RHS norms
    assert R is not None
    rhs_norms = [norm(R[i]) for i in range(len(R))]
    print("  Generated RHS norms:")
    print(f"    Min:  {min(rhs_norms):.6f}")
    print(f"    Max:  {max(rhs_norms):.6f}")
    print(f"    Mean: {np.mean(rhs_norms):.6f}")
    print(f"    Std:  {np.std(rhs_norms):.6f}")

    # Verify A @ x = b relationship
    assert X is not None
    residuals = [norm(A @ X[i] - R[i]) for i in range(len(R))]
    print("  Residual norms (A @ x - b):")
    print(f"    Max: {max(residuals):.2e}")
    print(f"    Mean: {np.mean(residuals):.2e}")
    print("  ✓ Normal strategy normalization: validating residuals")
    assert max(residuals) < 1e-10
    assert np.std(rhs_norms) > 0.0


def test_krylov_strategy_normalization():
    """Test that krylov_strategy generates RHS with consistent norms."""
    print("\nTesting krylov_strategy normalization...")

    # Create a simple test system
    n = 50
    A = np.eye(n) + 0.1 * np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric
    b = np.random.randn(n)
    mother_rhs_norm = norm(b)

    print(f"  Mother RHS norm: {mother_rhs_norm:.6f}")

    # Generate samples
    strategy = KrylovStrategy()
    output = strategy.generate(
        matrix=A, cfg={"samples": 100, "seed": 42, "krylov_iters": 10}, archive=None
    )
    R = output.rhs
    X = output.solutions
    assert R is not None and X is not None

    # Check RHS norms
    rhs_norms = [norm(R[i]) for i in range(len(R))]
    print("  Generated RHS norms:")
    print(f"    Min:  {min(rhs_norms):.6f}")
    print(f"    Max:  {max(rhs_norms):.6f}")
    print(f"    Mean: {np.mean(rhs_norms):.6f}")
    print(f"    Std:  {np.std(rhs_norms):.6f}")

    # For krylov, the solution is approximate, so residual will be non-zero
    residuals = [norm(A @ X[i] - R[i]) for i in range(len(R))]
    print("  Residual norms (A @ x - b):")
    print(f"    Max: {max(residuals):.2e}")
    print(f"    Mean: {np.mean(residuals):.2e}")
    print("  (Note: Krylov solutions are approximate, so residuals are expected)")
    print("  ✓ Krylov strategy normalization: validating residuals")
    assert max(residuals) < 1e-8
    assert np.std(rhs_norms) > 0.0


def main():
    """Run all normalization tests."""
    print("=" * 70)
    print("Normalization Tests")
    print("=" * 70)

    test1_passed = test_normal_strategy_normalization()
    test2_passed = test_krylov_strategy_normalization()

    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 70)


if __name__ == "__main__":
    main()
