"""Tests for unified solve configuration."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.helpers import _solve_linear_systems
from neuralls.domain.generation.matrix_operator import MatrixOperator
from neuralls.domain.generation.strategy_configs import SolveConfig


@pytest.fixture
def sample_spd_matrix() -> np.ndarray:
    """Create a small symmetric positive-definite matrix."""
    n = 20
    A = np.random.randn(n, n)
    return A.T @ A + 5 * np.eye(n)  # Well-conditioned SPD


@pytest.fixture
def sample_rhs(sample_spd_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create RHS vectors and exact solutions."""
    n = sample_spd_matrix.shape[0]
    num_systems = 5

    # Create exact solutions
    solutions = np.random.randn(num_systems, n)

    # Compute RHS: b = A @ x
    rhs_vectors = np.array([sample_spd_matrix @ x for x in solutions])

    return rhs_vectors, solutions


def test_solve_config_default() -> None:
    """Test default SolveConfig values."""
    cfg = SolveConfig(
        method="direct",
        rtol=1e-15,
        atol=0.0,
        max_iters=1_000_000,
        assume_pos_def=True,
    )

    assert cfg.method == "direct"
    assert cfg.rtol > 0.0  # Has a default value
    assert cfg.atol >= 0.0
    assert cfg.max_iters > 0
    assert cfg.assume_pos_def is True


def test_solve_config_cg() -> None:
    """Test CG-specific configuration."""
    cfg = SolveConfig(
        method="cg",
        rtol=1e-10,
        atol=1e-15,
        max_iters=1000,
        assume_pos_def=False,
    )

    assert cfg.method == "cg"
    assert cfg.rtol == 1e-10
    assert cfg.atol == 1e-15
    assert cfg.max_iters == 1000
    assert cfg.assume_pos_def is False


def test_solve_config_frozen() -> None:
    """Test that SolveConfig is frozen (immutable)."""
    from pydantic_core import ValidationError as PydanticValidationError

    cfg = SolveConfig(
        method="direct",
        rtol=1e-15,
        atol=0.0,
        max_iters=1_000_000,
        assume_pos_def=True,
    )

    with pytest.raises(PydanticValidationError, match="Instance is frozen"):
        cfg.method = "cg"  # type: ignore[misc]


def test_solve_linear_systems_direct(
    sample_spd_matrix: np.ndarray, sample_rhs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test direct solve produces accurate solutions."""
    rhs_vectors, exact_solutions = sample_rhs

    computed_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="direct",
        assume_pos_def=True,
    )

    # Check shape
    assert computed_solutions.shape == exact_solutions.shape

    # Check accuracy (should be very accurate with direct solve)
    for i in range(len(rhs_vectors)):
        residual = np.linalg.norm(sample_spd_matrix @ computed_solutions[i] - rhs_vectors[i])
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_error = residual / rhs_norm
        assert rel_error < 1e-10  # Very tight for direct solve


def test_solve_linear_systems_cg(
    sample_spd_matrix: np.ndarray, sample_rhs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test CG solve produces accurate solutions."""
    rhs_vectors, exact_solutions = sample_rhs

    computed_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="cg",
        rtol=1e-12,
        atol=0.0,
        max_iters=100,
    )

    # Check shape
    assert computed_solutions.shape == exact_solutions.shape

    # Check accuracy (CG should converge to high accuracy for well-conditioned SPD)
    for i in range(len(rhs_vectors)):
        residual = np.linalg.norm(sample_spd_matrix @ computed_solutions[i] - rhs_vectors[i])
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_error = residual / rhs_norm
        assert rel_error < 1e-8  # Looser tolerance for CG


def test_solve_linear_systems_invalid_method(
    sample_spd_matrix: np.ndarray, sample_rhs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that invalid method raises error."""
    rhs_vectors, _ = sample_rhs

    with pytest.raises(ValueError, match="Invalid solve method"):
        _solve_linear_systems(
            MatrixOperator(sample_spd_matrix),
            rhs_vectors,
            method="invalid",  # type: ignore[arg-type]
        )


def test_solve_linear_systems_direct_vs_cg_consistency(
    sample_spd_matrix: np.ndarray, sample_rhs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that direct and CG solves produce similar results."""
    rhs_vectors, _ = sample_rhs

    direct_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="direct",
        assume_pos_def=True,
    )

    cg_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="cg",
        rtol=1e-12,
        atol=0.0,
        max_iters=100,
    )

    # Solutions should be close
    for i in range(len(rhs_vectors)):
        diff = np.linalg.norm(direct_solutions[i] - cg_solutions[i])
        sol_norm = np.linalg.norm(direct_solutions[i])
        rel_diff = diff / sol_norm
        assert rel_diff < 1e-6  # CG and direct should agree closely


def test_solve_linear_systems_cg_tolerance_effect(
    sample_spd_matrix: np.ndarray, sample_rhs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that tighter tolerance produces more accurate solutions."""
    rhs_vectors, _ = sample_rhs

    # Loose tolerance
    loose_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="cg",
        rtol=1e-6,
        atol=0.0,
        max_iters=100,
    )

    # Tight tolerance
    tight_solutions = _solve_linear_systems(
        MatrixOperator(sample_spd_matrix),
        rhs_vectors,
        method="cg",
        rtol=1e-12,
        atol=0.0,
        max_iters=100,
    )

    # Compute residuals
    loose_errors = [
        np.linalg.norm(sample_spd_matrix @ sol - rhs) / np.linalg.norm(rhs)
        for sol, rhs in zip(loose_solutions, rhs_vectors)
    ]

    tight_errors = [
        np.linalg.norm(sample_spd_matrix @ sol - rhs) / np.linalg.norm(rhs)
        for sol, rhs in zip(tight_solutions, rhs_vectors)
    ]

    # Tight tolerance should produce smaller errors
    assert max(tight_errors) < max(loose_errors)


def test_eigenvector_inverse_with_solve_config() -> None:
    """Test EigenvectorInverseStrategy with custom solve config."""
    from neuralls.domain.generation.strategies.eigenvector import EigenvectorInverseStrategy

    # Create symmetric matrix
    n = 20
    A = np.random.randn(n, n)
    A = A.T @ A + 5 * np.eye(n)

    # Test with direct solve (default)
    cfg_direct = {
        "samples": 5,
        "which": "smallest",
        "num_eigenvectors": 5,
        "include_eigenvectors": True,
        "seed": 42,
        "solve_config": {
            "method": "direct",
            "assume_pos_def": True,
        },
    }

    strategy = EigenvectorInverseStrategy()
    samples_direct = strategy.generate(A, cfg=cfg_direct)
    assert samples_direct.solutions is not None
    assert samples_direct.rhs is not None

    assert samples_direct.rhs.shape == (5, n)
    assert samples_direct.solutions.shape == (5, n)

    # Verify A @ x = b
    for i in range(5):
        assert samples_direct.rhs is not None
        assert samples_direct.solutions is not None
        residual = np.linalg.norm(A @ samples_direct.solutions[i] - samples_direct.rhs[i])
        rhs_norm = np.linalg.norm(samples_direct.rhs[i])
        rel_error = residual / rhs_norm
        assert rel_error < 1e-10

    # Test with CG solve
    cfg_cg = {
        "samples": 5,
        "which": "smallest",
        "num_eigenvectors": 5,
        "include_eigenvectors": True,
        "seed": 42,
        "solve_config": {
            "method": "cg",
            "rtol": 1e-12,
            "atol": 0.0,
            "max_iters": 100,
        },
    }

    samples_cg = strategy.generate(A, cfg=cfg_cg)

    assert samples_cg.solutions is not None

    # Verify A @ x = b (looser tolerance for CG)
    for i in range(5):
        assert samples_cg.rhs is not None
        residual = np.linalg.norm(A @ samples_cg.solutions[i] - samples_cg.rhs[i])
        rhs_norm = np.linalg.norm(samples_cg.rhs[i])
        rel_error = residual / rhs_norm
        assert rel_error < 1e-8  # Looser for CG


def test_solve_config_pydantic_validation() -> None:
    """Test that invalid SolveConfig values are rejected."""
    from pydantic import ValidationError as PydanticValidationError

    # Valid config
    SolveConfig(
        method="direct",
        rtol=1e-12,
        atol=0.0,
        max_iters=1_000_000,
        assume_pos_def=True,
    )

    # Invalid method
    with pytest.raises(PydanticValidationError):
        SolveConfig.model_validate(
            {
                "method": "invalid",
                "rtol": 1e-15,
                "atol": 0.0,
                "max_iters": 1_000_000,
                "assume_pos_def": True,
            }
        )

    # Negative rtol
    with pytest.raises(PydanticValidationError):
        SolveConfig(
            method="direct",
            rtol=-1.0,
            atol=0.0,
            max_iters=1_000_000,
            assume_pos_def=True,
        )

    # Negative atol
    with pytest.raises(PydanticValidationError):
        SolveConfig(
            method="direct",
            rtol=1e-15,
            atol=-1.0,
            max_iters=1_000_000,
            assume_pos_def=True,
        )

    # Invalid max_iters (zero)
    with pytest.raises(PydanticValidationError):
        SolveConfig(
            method="direct",
            rtol=1e-15,
            atol=0.0,
            max_iters=0,
            assume_pos_def=True,
        )

    # max_iters too large (exceeds MAX_ITERATIONS_UPPER_LIMIT)
    with pytest.raises(PydanticValidationError):
        SolveConfig(
            method="direct",
            rtol=1e-15,
            atol=0.0,
            max_iters=10_000_000,
            assume_pos_def=True,
        )
