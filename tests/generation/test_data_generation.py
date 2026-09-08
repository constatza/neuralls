from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from neuralls.domain.generation import generate_mixture
from neuralls.domain.generation.helpers import (
    _generate_eigenvector_combinations,
)
from neuralls.domain.generation.orchestration import _shuffle_samples
from neuralls.domain.normalization import ErrorTraceSamples

# ========================================================================
# Tests for residuals strategy
# ========================================================================


def test_error_strategy_with_random(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Residuals strategy embeds trace pairs (r_k, e_k) directly into rhs/solutions.

    A @ e_k = r_k must hold for every row.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        archive_solutions: Pre-computed archive solutions fixture
        archive_rhs: Pre-computed archive RHS vectors fixture
        test_seed: Random seed fixture
        solver_overrides: Default tracing solvers for single-RHS strategies
    """
    cg_iters = 3
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"residuals": 1.0},
        total=2,
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    expected_rows = 2
    assert rhs.shape == (expected_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    assert residuals is None
    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_error_strategy_with_archive(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Residuals strategy with archive produces trace rows satisfying A @ sol = rhs."""
    cg_iters = 3
    rhs, solutions, _residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"residuals": 1.0},
        total=2,
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    expected_rows = 2
    assert rhs.shape == (expected_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_error_vectors_satisfy_equation(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """A @ e_k = r_k holds for all trace rows (A e_k = r_k by construction)."""
    cg_iters = 5
    rhs, solutions, _, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"residuals": 1.0},
        total=3,
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_residuals_match_current_solutions(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """A @ solutions = rhs holds for all rows (r_k = A @ e_k by construction)."""
    rhs, solutions, _, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"residuals": 1.0},
        total=2,
        strategy_overrides={"residuals": {"cg_iters": 4}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_error_strategy_mixed_with_forward_strategy(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Mixing neutral_ones + residuals concatenates rows; A @ sol = rhs for all."""
    cg_iters = 2
    rhs, solutions, _residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        counts={"neutral_ones": 1, "residuals": 6},
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    residual_rows = 6
    assert rhs.shape == (1 + residual_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_shuffle_samples_reindexes_error_traces() -> None:
    """Shuffling must keep residual-error trace references aligned with base systems."""
    rng = np.random.default_rng(0)
    X = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    Y = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    error_traces = ErrorTraceSamples(
        residuals=np.array([[100.0], [101.0], [200.0], [300.0]], dtype=np.float64),
        solutions_current=np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64),
        errors=np.array([[0.9], [0.8], [1.7], [2.6]], dtype=np.float64),
        true_solutions=Y.copy(),
        sample_indices=np.array([0, 0, 1, 2], dtype=np.int64),
        iteration_indices=np.array([0, 1, 0, 0], dtype=np.int64),
    )
    expected_order = np.random.default_rng(0).permutation(len(X))
    inverse = np.empty_like(expected_order)
    inverse[expected_order] = np.arange(len(expected_order))

    shuffled = _shuffle_samples(
        X,
        Y,
        None,
        error_traces,
        np.zeros(len(X), dtype=np.uint8),
        rng.permutation(len(X)),
    )

    shuffled_error_traces = shuffled.error_traces
    assert shuffled_error_traces is not None
    expected_indices = inverse[error_traces.sample_indices]
    expected_true_solutions = Y[expected_order]

    np.testing.assert_allclose(shuffled.rhs, X[expected_order])
    np.testing.assert_allclose(shuffled.solutions, expected_true_solutions)
    np.testing.assert_array_equal(shuffled_error_traces.sample_indices, expected_indices)
    np.testing.assert_allclose(
        shuffled_error_traces.true_solutions,
        expected_true_solutions,
    )


def test_error_strategy_validation(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Test error strategy validates sufficient archive samples.

    Verifies that the strategy raises an appropriate error when
    the archive doesn't contain enough solutions.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
        solver_overrides: Default tracing solvers for single-RHS strategies
    """
    # Archive with only 1 solution, but we need 2 base systems after integer division
    insufficient_archive = np.array([[0.5, 0.3]], dtype=np.float64)

    try:
        _rhs, _solutions, _residuals, _error_traces = generate_mixture(
            A=small_spd_matrix,
            mix={"residuals": 1.0},
            total=8,
            strategy_overrides={"residuals": {"cg_iters": 3}},
            solver_overrides=solver_overrides,
            seed=test_seed,
            shuffle=False,
            archive_solutions=insufficient_archive,
        )
        assert False, "Should have raised ValueError for insufficient archive"
    except ValueError as e:
        assert "Not enough archive lhs" in str(e), (
            "Error message should mention insufficient archive lhs vectors"
        )


def test_error_strategy_in_generate_mixture(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Mixing normal + residuals produces concatenated rows satisfying A @ sol = rhs."""
    cg_iters = 3
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"normal": 0.5, "residuals": 0.5},
        total=4,
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    normal_rows = 2
    residual_rows = 2
    assert rhs.shape == (normal_rows + residual_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    assert error_traces is None
    assert residuals is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_error_strategy_traces_structure(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Residuals strategy produces base_systems * (cg_iters+1) rows total."""
    cg_iters = 4
    rhs, solutions, _, error_traces = generate_mixture(
        A=small_spd_matrix,
        mix={"residuals": 1.0},
        total=3,
        strategy_overrides={"residuals": {"cg_iters": cg_iters}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    expected_rows = 3
    assert rhs.shape == (expected_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    assert error_traces is None
    np.testing.assert_allclose(
        (small_spd_matrix @ solutions.T).T,
        rhs,
        atol=1e-10,
    )


def test_error_strategy_with_zero_iterations(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Test error strategy rejects zero CG iterations.

    Verifies that Pydantic validation rejects cg_iters=0.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
        solver_overrides: Default tracing solvers for single-RHS strategies
    """
    with pytest.raises(ValueError, match="(greater than or equal to 1|cg_iters)"):
        _rhs, _solutions, _residuals, _error_traces = generate_mixture(
            A=small_spd_matrix,
            mix={"residuals": 1.0},
            total=2,
            strategy_overrides={"residuals": {"cg_iters": 0}},  # Zero iterations
            solver_overrides=solver_overrides,
            seed=test_seed,
            shuffle=False,
        )


# =============================================================================
# EIGENVECTOR STRATEGY TESTS
# =============================================================================


def test_eigenvector_forward_basic(tmp_path: Path) -> None:
    """Test eigenvector forward strategy produces exact samples."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=2,
        seed=42,
        shuffle=False,
    )

    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    # Verify b = A @ x for each sample (machine precision)
    for i in range(2):
        residual = rhs[i] - A @ solutions[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_inverse_machine_precision(tmp_path: Path) -> None:
    """Test eigenvector inverse strategy achieves machine precision."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_inverse": 1.0},
        total=2,
        seed=42,
        shuffle=False,
    )

    # Verify A @ x = b at machine precision
    for i in range(2):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_requires_symmetric(tmp_path: Path) -> None:
    """Test eigenvector strategies reject non-symmetric matrices."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)  # Asymmetric
    np.array([1.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric"):
        generate_mixture(A=A, mix={"eigenvector_forward": 1.0}, total=2)


def test_eigenvector_count_exceeds_dimension(tmp_path: Path) -> None:
    """Test that requesting more samples than eigenvectors now works via combinations."""
    A = np.eye(3, dtype=np.float64) * 2.0
    np.ones(3, dtype=np.float64)

    # With new implementation, this should work (generates 5 combinations from 3 eigenvectors)
    rhs, solutions, _, _ = generate_mixture(A=A, mix={"eigenvector_forward": 1.0}, total=5)

    assert rhs.shape == (5, 3)
    assert solutions.shape == (5, 3)

    # Verify accuracy
    for i in range(5):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_selection_modes(tmp_path: Path) -> None:
    """Test different eigenvector eigenvalue range selections."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    np.ones(4, dtype=np.float64)

    # Test "smallest" mode - use eigenvectors with k smallest eigenvalues
    _, sols_first, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "which": "smallest",
                "num_eigenvectors": 2,  # Use k=2 smallest eigenvalues
            }
        },
        shuffle=False,
        seed=42,
    )

    # Test "largest" mode - use eigenvectors with k largest eigenvalues
    _, sols_last, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "which": "largest",
                "num_eigenvectors": 2,  # Use k=2 largest eigenvalues
            }
        },
        shuffle=False,
        seed=42,
    )

    # Solutions should be different (different eigenvector subspaces)
    assert not np.allclose(sols_first, sols_last)


def test_mixed_strategy_with_eigenvectors(tmp_path: Path) -> None:
    """Test mixing random and eigenvector strategies."""
    A = np.eye(10, dtype=np.float64) * 2.0
    np.ones(10, dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"random": 0.5, "eigenvector_forward": 0.5},
        total=20,
        seed=42,
    )

    assert rhs.shape == (20, 10)
    assert solutions.shape == (20, 10)

    # All samples should have low residuals
    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-9, f"Sample {i}: residual {rel_residual:.2e}"


# =============================================================================
# EIGENVECTOR LINEAR COMBINATIONS: Tests for random linear combinations
# =============================================================================


def test_generate_eigenvector_combinations_vectorized(tmp_path: Path) -> None:
    """Test vectorized linear combination generation."""
    # Use 4x4 diagonal matrix for simple eigenvectors
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    _eigenvalues, eigenvectors = eigh(A)

    # Select first 3 eigenvectors
    selected = eigenvectors[:, :3]  # Shape (4, 3)

    rng = np.random.default_rng(42)
    combinations = _generate_eigenvector_combinations(selected, num_samples=10, rng=rng)

    # Shape check
    assert combinations.shape == (10, 4), "Output shape mismatch"

    # Each combination should be in span of selected eigenvectors
    # Verify by projecting back onto basis
    for i in range(10):
        # Project onto basis and reconstruct
        reconstruction = selected @ (selected.T @ combinations[i])
        np.testing.assert_allclose(reconstruction, combinations[i], rtol=1e-10)

    # Verify linear independence (combinations should differ)
    for i in range(9):
        assert not np.allclose(combinations[i], combinations[i + 1])


def test_eigenvector_combinations_reproducible(tmp_path: Path) -> None:
    """Test that combinations are reproducible with same seed."""
    A = np.eye(5, dtype=np.float64) * 2.0
    _eigenvalues, eigenvectors = eigh(A)

    rng1 = np.random.default_rng(123)
    combinations1 = _generate_eigenvector_combinations(eigenvectors, 5, rng1)

    rng2 = np.random.default_rng(123)
    combinations2 = _generate_eigenvector_combinations(eigenvectors, 5, rng2)

    np.testing.assert_array_equal(combinations1, combinations2)


def test_eigenvector_forward_with_combinations(tmp_path: Path) -> None:
    """Test eigenvector_forward with linear combinations."""
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    np.ones(5, dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=20,  # Generate 20 samples
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 3,  # From basis of 3
                "which": "smallest",
                "include_eigenvectors": False,  # Only combinations
            }
        },
        seed=42,
        shuffle=False,
    )

    assert rhs.shape == (20, 5)
    assert solutions.shape == (20, 5)

    # Verify machine precision accuracy
    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_forward_include_eigenvectors(tmp_path: Path) -> None:
    """Test including original eigenvectors in output."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    np.ones(4, dtype=np.float64)

    _rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=6,  # 2 eigenvectors + 4 combinations
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "which": "smallest",
                "include_eigenvectors": True,
            }
        },
        seed=42,
        shuffle=False,
    )

    assert solutions.shape == (6, 4)

    # First 2 should be eigenvectors (diagonal matrix -> standard basis)
    _eigenvalues, eigenvectors = eigh(A)
    np.testing.assert_allclose(solutions[:2], eigenvectors[:, :2].T, rtol=1e-10)

    # Remaining 4 should be combinations (different from eigenvectors)
    for i in range(2, 6):
        # Should not match any single eigenvector exactly
        for j in range(4):
            assert not np.allclose(solutions[i], eigenvectors[:, j])


def test_eigenvector_inverse_with_combinations(tmp_path: Path) -> None:
    """Test eigenvector_inverse with combinations as RHS."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_inverse": 1.0},
        total=10,
        strategy_overrides={
            "eigenvector_inverse": {
                "num_eigenvectors": 2,  # Use both eigenvectors
                "include_eigenvectors": False,
            }
        },
        seed=42,
    )

    # Verify solutions are accurate
    for i in range(10):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_validation_num_exceeds_dimension(tmp_path: Path) -> None:
    """Test error when num_eigenvectors > matrix dimension."""
    A = np.eye(3, dtype=np.float64) * 2.0
    np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError, match="must be positive and ≤"):
        generate_mixture(
            A=A,
            mix={"eigenvector_forward": 1.0},
            total=5,
            strategy_overrides={"eigenvector_forward": {"num_eigenvectors": 5}},  # > 3
        )


def test_eigenvector_validation_include_requires_enough_samples(tmp_path: Path) -> None:
    """Test error when include_eigenvectors=True but samples < num_eigenvectors."""
    A = np.eye(5, dtype=np.float64) * 2.0
    np.ones(5, dtype=np.float64)

    with pytest.raises(ValueError, match="must be >="):
        generate_mixture(
            A=A,
            mix={"eigenvector_forward": 1.0},
            total=2,  # Only 2 samples
            strategy_overrides={
                "eigenvector_forward": {
                    "num_eigenvectors": 3,  # But need space for 3 eigenvectors
                    "include_eigenvectors": True,
                }
            },
        )


def test_eigenvector_forward_rhs_computation_change(tmp_path: Path) -> None:
    """Verify RHS is computed as A @ v (not λ * v) for forward strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "include_eigenvectors": False,
            }
        },
        seed=42,
        shuffle=False,
    )

    # Verify b = A @ x directly (not using eigenvalue equation)
    for i in range(2):
        expected_rhs = A @ solutions[i]
        np.testing.assert_allclose(rhs[i], expected_rhs, rtol=1e-14)


def test_eigenvector_backward_compatible_defaults(tmp_path: Path) -> None:
    """Test that old configs still work without new parameters."""
    A = np.diag([1.0, 2.0, 3.0])
    np.ones(3, dtype=np.float64)

    # Old-style config (no new parameters)
    rhs, solutions, _, _ = generate_mixture(
        A=A,
        mix={"eigenvector_forward": 1.0},
        total=3,
        strategy_overrides={"eigenvector_forward": {}},
        seed=42,
    )

    # Should generate 3 combinations from all 3 eigenvectors
    assert rhs.shape == (3, 3)
    assert solutions.shape == (3, 3)

    # Verify accuracy
    for i in range(3):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14

    # =============================================================================
    # PYDANTIC VALIDATION TESTS
    # =============================================================================

    def test_pydantic_rejects_unknown_parameters() -> None:
        """Test that Pydantic validation rejects unknown parameters due to extra='forbid'.

        Note: The orchestration layer filters out unknown parameters before passing to strategies,
        so this test validates by directly instantiating the config class.
        """
        from pydantic import ValidationError

        from neuralls.domain.generation.strategy_configs import KrylovConfig

        # Try to create a config with an unknown parameter directly
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            KrylovConfig.model_validate({"samples": 10, "unknown_param": 123})


def test_pydantic_rejects_invalid_literal_values() -> None:
    """Test that Pydantic validation rejects invalid Literal values."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass an invalid 'which' value (should be 'smallest', 'largest', or 'both')
    with pytest.raises(ValueError, match="Input should be"):
        generate_mixture(
            A=A,
            mix={"eigenvector_forward": 1.0},
            total=2,
            strategy_overrides={"eigenvector_forward": {"which": "invalid"}},
        )


def test_pydantic_requires_rhs_glob_for_rhs_archive() -> None:
    """Test that rhs_glob is required for rhs_archive strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    # Try to use rhs_archive without providing rhs_glob
    with pytest.raises(ValueError, match="Field required"):
        generate_mixture(
            A=A,
            mix={"rhs_archive": 1.0},
            total=2,
            strategy_overrides={"rhs_archive": {}},  # Missing rhs_glob
        )


def test_pydantic_requires_solutions_glob_for_solution_archive() -> None:
    """Test that solutions_glob is required for solution_archive strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    # Try to use solution_archive without providing solutions_glob
    with pytest.raises(ValueError, match="Field required"):
        generate_mixture(
            A=A,
            mix={"solution_archive": 1.0},
            total=2,
            strategy_overrides={"solution_archive": {}},  # Missing solutions_glob
        )


def test_pydantic_validates_residual_iters_type(solver_overrides: dict) -> None:
    """Test that Pydantic validates parameter types (cg_iters must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass a string for cg_iters (should be int)
    with pytest.raises(ValueError, match="Input should be a valid integer"):
        generate_mixture(
            A=A,
            mix={"residuals": 1.0},
            total=2,
            strategy_overrides={"residuals": {"cg_iters": "many"}},
            solver_overrides=solver_overrides,
        )


def test_pydantic_validates_krylov_iters_type() -> None:
    """Test that Pydantic validates parameter types (krylov_iters must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass a float for krylov_iters (should be int)
    with pytest.raises(ValueError):
        generate_mixture(
            A=A,
            mix={"krylov": 1.0},
            total=2,
            strategy_overrides={"krylov": {"krylov_iters": 15.5}},
        )
