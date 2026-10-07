from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg import eigh

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.helpers import (
    _generate_eigenvector_combinations,
)
from neuralls.domain.generation.interfaces import ArchiveData, TracingSolverCallable
from neuralls.domain.generation.runner import GeneratedSamples
from neuralls.domain.generation.strategy_configs import KrylovConfig


def _pairs(generated: GeneratedSamples) -> tuple[np.ndarray, np.ndarray]:
    """The (rhs, solutions) training pairs one strategy emits, trace rows included."""
    if generated.error_traces is not None:
        return generated.error_traces.residuals, generated.error_traces.errors
    if generated.residual_traces is not None:
        return generated.residual_traces.residuals, generated.residual_traces.solutions
    assert generated.rhs is not None and generated.solutions is not None
    assert generated.rhs.ndim == 2 and generated.solutions.ndim == 2
    return generated.rhs, generated.solutions


def _generate(
    matrix: np.ndarray,
    counts: dict[str, int],
    *,
    seed: int = 42,
    overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run each strategy for its count and stack the rows in mixture order.

    Returns:
        ``(rhs, solutions)``, each of shape (rows, n).
    """
    archive = (
        ArchiveData(lhs=archive_solutions, rhs=archive_rhs)
        if archive_solutions is not None
        else None
    )
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    for name, count in counts.items():
        if count == 0:
            continue
        cfg = {**(overrides or {}).get(name, {}), "samples": count, "seed": seed}
        generated = run_generation(
            name,
            matrix,
            cfg=cfg,
            solver=(solver_overrides or {}).get(name),
            archive=archive,
        )
        rhs, solutions = _pairs(generated)
        rhs_blocks.append(rhs)
        solution_blocks.append(solutions)
    return np.vstack(rhs_blocks), np.vstack(solution_blocks)


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
    """
    rhs, solutions = _generate(
        small_spd_matrix,
        {"residuals": 2},
        overrides={"residuals": {"stop": 3, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert rhs.shape == (2, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_with_archive(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Residuals strategy with archive produces trace rows satisfying A @ sol = rhs."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"residuals": 2},
        overrides={"residuals": {"stop": 3, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert rhs.shape == (2, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_vectors_satisfy_equation(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """A @ e_k = r_k holds for all trace rows (A e_k = r_k by construction)."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"residuals": 3},
        overrides={"residuals": {"stop": 2, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_residuals_match_current_solutions(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """A @ solutions = rhs holds for all rows (r_k = A @ e_k by construction)."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"residuals": 2},
        overrides={"residuals": {"stop": 4, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_mixed_with_forward_strategy(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Mixing neutral_ones + residuals concatenates rows; A @ sol = rhs for all."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"neutral_ones": 1, "residuals": 6},
        overrides={"residuals": {"stop": 2, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    residual_rows = 6
    assert rhs.shape == (1 + residual_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_validation(
    small_spd_matrix: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """The strategy rejects an archive with too few solutions for its base systems."""
    insufficient_archive = np.array([[0.5, 0.3]], dtype=np.float64)

    with pytest.raises(ValueError, match="Not enough archive lhs"):
        _generate(
            small_spd_matrix,
            {"residuals": 8},
            overrides={"residuals": {"stop": 3, "start": 1}},
            solver_overrides=solver_overrides,
            seed=test_seed,
            archive_solutions=insufficient_archive,
        )


def test_error_strategy_mixed_with_generated_strategy(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Mixing normal + residuals produces concatenated rows satisfying A @ sol = rhs."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"normal": 2, "residuals": 2},
        overrides={"residuals": {"stop": 3, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    normal_rows = 2
    residual_rows = 2
    assert rhs.shape == (normal_rows + residual_rows, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_traces_structure(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Residuals strategy produces one row per requested sample, trimmed from base systems."""
    rhs, solutions = _generate(
        small_spd_matrix,
        {"residuals": 3},
        overrides={"residuals": {"stop": 4, "start": 1}},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert rhs.shape == (3, small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_with_zero_iterations(
    small_spd_matrix: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """Pydantic validation rejects stop=0 for the residuals strategy."""
    with pytest.raises(ValueError, match="(greater than or equal to 1|stop)"):
        _generate(
            small_spd_matrix,
            {"residuals": 2},
            overrides={"residuals": {"stop": 0}},
            solver_overrides=solver_overrides,
            seed=test_seed,
        )


# =============================================================================
# EIGENVECTOR STRATEGY TESTS
# =============================================================================


def test_eigenvector_forward_basic() -> None:
    """Eigenvector forward strategy produces exact samples."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    rhs, solutions = _generate(A, {"eigenvector_forward": 2})

    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    # Verify b = A @ x for each sample (machine precision)
    for i in range(2):
        residual = rhs[i] - A @ solutions[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_inverse_machine_precision() -> None:
    """Eigenvector inverse strategy achieves machine precision."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    rhs, solutions = _generate(A, {"eigenvector_inverse": 2})

    # Verify A @ x = b at machine precision
    for i in range(2):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_requires_symmetric() -> None:
    """Eigenvector strategies reject non-symmetric matrices."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)  # Asymmetric

    with pytest.raises(ValueError, match="symmetric"):
        _generate(A, {"eigenvector_forward": 2})


def test_eigenvector_count_exceeds_dimension() -> None:
    """Requesting more samples than eigenvectors works via combinations."""
    A = np.eye(3, dtype=np.float64) * 2.0

    rhs, solutions = _generate(A, {"eigenvector_forward": 5})

    assert rhs.shape == (5, 3)
    assert solutions.shape == (5, 3)

    for i in range(5):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_selection_modes() -> None:
    """Different eigenvalue range selections give different eigenvector subspaces."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])

    _, sols_first = _generate(
        A,
        {"eigenvector_forward": 2},
        overrides={"eigenvector_forward": {"which": "smallest", "num_eigenvectors": 2}},
    )
    _, sols_last = _generate(
        A,
        {"eigenvector_forward": 2},
        overrides={"eigenvector_forward": {"which": "largest", "num_eigenvectors": 2}},
    )

    assert not np.allclose(sols_first, sols_last)


def test_mixed_strategy_with_eigenvectors() -> None:
    """Mixing random and eigenvector strategies keeps every sample accurate."""
    A = np.eye(10, dtype=np.float64) * 2.0

    rhs, solutions = _generate(A, {"random": 10, "eigenvector_forward": 10})

    assert rhs.shape == (20, 10)
    assert solutions.shape == (20, 10)

    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-9, f"Sample {i}: residual {rel_residual:.2e}"


# =============================================================================
# EIGENVECTOR LINEAR COMBINATIONS: Tests for random linear combinations
# =============================================================================


def test_generate_eigenvector_combinations_vectorized() -> None:
    """Vectorized linear combination generation."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    _eigenvalues, eigenvectors = eigh(A)

    # Select first 3 eigenvectors
    selected = eigenvectors[:, :3]  # Shape (4, 3)

    rng = np.random.default_rng(42)
    combinations = _generate_eigenvector_combinations(selected, num_samples=10, rng=rng)

    assert combinations.shape == (10, 4), "Output shape mismatch"

    # Each combination should be in span of selected eigenvectors
    for i in range(10):
        reconstruction = selected @ (selected.T @ combinations[i])
        np.testing.assert_allclose(reconstruction, combinations[i], rtol=1e-10)

    # Combinations should differ from one another
    for i in range(9):
        assert not np.allclose(combinations[i], combinations[i + 1])


def test_eigenvector_combinations_reproducible() -> None:
    """Combinations are reproducible with the same seed."""
    A = np.eye(5, dtype=np.float64) * 2.0
    _eigenvalues, eigenvectors = eigh(A)

    rng1 = np.random.default_rng(123)
    combinations1 = _generate_eigenvector_combinations(eigenvectors, 5, rng1)

    rng2 = np.random.default_rng(123)
    combinations2 = _generate_eigenvector_combinations(eigenvectors, 5, rng2)

    np.testing.assert_array_equal(combinations1, combinations2)


def test_eigenvector_forward_with_combinations() -> None:
    """eigenvector_forward with linear combinations only."""
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])

    rhs, solutions = _generate(
        A,
        {"eigenvector_forward": 20},
        overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 3,
                "which": "smallest",
                "include_eigenvectors": False,
            }
        },
    )

    assert rhs.shape == (20, 5)
    assert solutions.shape == (20, 5)

    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_forward_include_eigenvectors() -> None:
    """Including original eigenvectors puts them first, then combinations."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])

    _rhs, solutions = _generate(
        A,
        {"eigenvector_forward": 6},  # 2 eigenvectors + 4 combinations
        overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "which": "smallest",
                "include_eigenvectors": True,
            }
        },
    )

    assert solutions.shape == (6, 4)

    # First 2 should be eigenvectors (diagonal matrix -> standard basis)
    _eigenvalues, eigenvectors = eigh(A)
    np.testing.assert_allclose(solutions[:2], eigenvectors[:, :2].T, rtol=1e-10)

    # Remaining 4 should be combinations (not any single eigenvector)
    for i in range(2, 6):
        for j in range(4):
            assert not np.allclose(solutions[i], eigenvectors[:, j])


def test_eigenvector_inverse_with_combinations() -> None:
    """eigenvector_inverse with combinations as RHS."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    rhs, solutions = _generate(
        A,
        {"eigenvector_inverse": 10},
        overrides={
            "eigenvector_inverse": {
                "num_eigenvectors": 2,  # Use both eigenvectors
                "include_eigenvectors": False,
            }
        },
    )

    for i in range(10):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_validation_num_exceeds_dimension() -> None:
    """Error when num_eigenvectors > matrix dimension."""
    A = np.eye(3, dtype=np.float64) * 2.0

    with pytest.raises(ValueError, match="must be positive and ≤"):
        _generate(
            A,
            {"eigenvector_forward": 5},
            overrides={"eigenvector_forward": {"num_eigenvectors": 5}},
        )


def test_eigenvector_validation_include_requires_enough_samples() -> None:
    """Error when include_eigenvectors=True but samples < num_eigenvectors."""
    A = np.eye(5, dtype=np.float64) * 2.0

    with pytest.raises(ValueError, match="must be >="):
        _generate(
            A,
            {"eigenvector_forward": 2},  # Only 2 samples
            overrides={
                "eigenvector_forward": {
                    "num_eigenvectors": 3,  # But need space for 3 eigenvectors
                    "include_eigenvectors": True,
                }
            },
        )


def test_eigenvector_forward_rhs_computation_change() -> None:
    """RHS is computed as A @ v (not lambda * v) for the forward strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    rhs, solutions = _generate(
        A,
        {"eigenvector_forward": 2},
        overrides={"eigenvector_forward": {"num_eigenvectors": 2, "include_eigenvectors": False}},
    )

    for i in range(2):
        expected_rhs = A @ solutions[i]
        np.testing.assert_allclose(rhs[i], expected_rhs, rtol=1e-14)


def test_eigenvector_backward_compatible_defaults() -> None:
    """Old-style configs still work without the new parameters."""
    A = np.diag([1.0, 2.0, 3.0])

    rhs, solutions = _generate(A, {"eigenvector_forward": 3}, overrides={"eigenvector_forward": {}})

    # Should generate 3 combinations from all 3 eigenvectors
    assert rhs.shape == (3, 3)
    assert solutions.shape == (3, 3)

    for i in range(3):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


# =============================================================================
# PYDANTIC VALIDATION TESTS
# =============================================================================


def test_pydantic_rejects_unknown_parameters() -> None:
    """Pydantic validation rejects unknown parameters (extra='forbid').

    The orchestration layer does not filter unknown keys, so the config class is
    validated directly.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        KrylovConfig.model_validate({"samples": 10, "unknown_param": 123})


def test_pydantic_rejects_invalid_literal_values() -> None:
    """Pydantic validation rejects invalid Literal values."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Input should be"):
        _generate(
            A, {"eigenvector_forward": 2}, overrides={"eigenvector_forward": {"which": "invalid"}}
        )


def test_pydantic_requires_rhs_glob_for_rhs_archive() -> None:
    """rhs_glob is required for rhs_archive."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Field required"):
        _generate(A, {"rhs_archive": 2}, overrides={"rhs_archive": {}})


def test_pydantic_requires_solutions_glob_for_solution_archive() -> None:
    """solutions_glob is required for solution_archive."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Field required"):
        _generate(A, {"solution_archive": 2}, overrides={"solution_archive": {}})


def test_pydantic_validates_residual_iters_type(solver_overrides: dict) -> None:
    """Pydantic validates parameter types (stop must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Input should be a valid integer"):
        _generate(
            A,
            {"residuals": 2},
            overrides={"residuals": {"stop": "many"}},
            solver_overrides=solver_overrides,
        )


def test_pydantic_validates_krylov_iters_type() -> None:
    """Pydantic validates parameter types (krylov_iters must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        _generate(A, {"krylov": 2}, overrides={"krylov": {"krylov_iters": 15.5}})
