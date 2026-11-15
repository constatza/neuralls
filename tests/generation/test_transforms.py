"""Tests for transformation layer (Layer 2: Pure Computation)."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.strategy_configs import SolveConfig
from neuralls.domain.generation.transforms import (
    ComputeRhsTransform,
    EigenvectorCombinationTransform,
    SolveTransform,
    VerifyTransform,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator for every random input of one test."""
    return np.random.default_rng(0)


@pytest.fixture
def sample_spd_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample 10x10 symmetric positive-definite matrix."""
    A = rng.standard_normal((10, 10))
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
    transform = SolveTransform(diagonal_matrix, SolveConfig(method="direct"))
    b = np.ones((5, 10)) * 2.0

    x = transform.transform(b).solutions

    assert x.shape == (5, 10)
    assert x.dtype == np.float64
    # For diagonal matrix: x = b / 2
    assert np.allclose(x, np.ones((5, 10)))


def test_solve_transform_inverse_cg(
    sample_spd_matrix: np.ndarray, rng: np.random.Generator
) -> None:
    """Test inverse transform with CG solve."""
    transform = SolveTransform(
        sample_spd_matrix, SolveConfig(method="cg", rtol=1e-12, max_iters=500)
    )

    # Create RHS from known solution
    x_true = rng.standard_normal((5, 10))
    b = np.array([sample_spd_matrix @ x for x in x_true])

    solved = transform.transform(b)
    x_solved = solved.solutions

    assert solved.unconverged == ()
    assert x_solved.shape == (5, 10)
    # Check that solved x is close to true x
    for i in range(5):
        residual = sample_spd_matrix @ x_solved[i] - b[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(b[i])
        assert rel_residual < 1e-10


def test_solve_transform_reports_rows_cg_did_not_converge_on(
    sample_spd_matrix: np.ndarray,
) -> None:
    """One CG iteration cannot solve a dense 10x10 system, so every row is reported."""
    transform = SolveTransform(sample_spd_matrix, SolveConfig(method="cg", max_iters=1))
    b = np.ones((3, 10))

    solved = transform.transform(b)

    assert [notice.row for notice in solved.unconverged] == [0, 1, 2]
    assert all(notice.exit_code == 1 for notice in solved.unconverged)


def test_verify_transform_accurate_solutions(
    sample_spd_matrix: np.ndarray, rng: np.random.Generator
) -> None:
    """Test verification transform with accurate solutions."""
    transform = VerifyTransform(sample_spd_matrix)

    x = rng.standard_normal((5, 10))
    b = np.array([sample_spd_matrix @ xi for xi in x])

    residuals = transform.transform((x, b))

    assert residuals.shape == (5,)
    assert np.all(residuals < 1e-10)


def test_verify_transform_inaccurate_solutions(
    sample_spd_matrix: np.ndarray, rng: np.random.Generator
) -> None:
    """Test verification transform measures inaccurate solutions."""
    transform = VerifyTransform(sample_spd_matrix)

    x = rng.standard_normal((5, 10))
    b_wrong = rng.standard_normal((5, 10))  # Random RHS, not A @ x

    residuals = transform.transform((x, b_wrong))

    assert residuals.shape == (5,)
    # Residuals should be large
    assert np.all(residuals > 0.1)


@pytest.mark.parametrize(
    ("count", "which"),
    [
        pytest.param(5, "smallest", id="smallest"),
        pytest.param(3, "largest", id="largest"),
        pytest.param(7, "random", id="random"),
    ],
)
def test_eigenvector_combination_transform_yields_unit_vectors(
    sample_spd_matrix: np.ndarray, count: int, which: str
) -> None:
    transform = EigenvectorCombinationTransform(
        count=count, which=which, rng=np.random.default_rng(42)
    )

    x = transform.transform(sample_spd_matrix)

    assert x.shape == (count, 10)
    assert x.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-10)


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
    inverse_transform = SolveTransform(sample_spd_matrix, SolveConfig(method="cg", rtol=1e-12))
    verify_transform = VerifyTransform(sample_spd_matrix)

    # Execute pipeline
    x_original = eigen_transform.transform(sample_spd_matrix)
    b = forward_transform.transform(x_original)
    x_recovered = inverse_transform.transform(b).solutions
    residuals = verify_transform.transform((x_recovered, b))

    # Verify pipeline correctness
    assert np.all(residuals < 1e-10)
    assert np.allclose(x_original, x_recovered, rtol=1e-8, atol=1e-10)
