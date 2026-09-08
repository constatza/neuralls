"""Test that data generation maintains A @ x = b consistency after normalization.

This test verifies that the orchestration layer does not double-normalize data
returned by strategies. Strategies receive a normalized matrix and produce
normalized data (RHS and solutions). The orchestration layer should NOT apply
additional normalization to this data.

Regression test for bug where build_dataset() was applying scale.scale_rhs()
and scale.scale_solution() to data that was already normalized by strategies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import (
    load_dense_training_arrays,
    load_matrix_dense_sample,
)


@pytest.fixture
def matrix_path(tmp_path: Path) -> str:
    """Create test matrix and return its path."""
    # Create a small SPD matrix for testing
    n = 20
    A = np.random.RandomState(42).randn(n, n)
    A = A.T @ A + np.eye(n) * 10.0

    matrix_file = tmp_path / "test_matrix.txt"
    np.savetxt(matrix_file, A)
    return str(matrix_file)


@pytest.fixture
def solutions_glob(tmp_path: Path, matrix_path: str) -> str:
    """Create solution archive files and return glob pattern."""
    # Load the matrix
    A = np.loadtxt(matrix_path)
    n = A.shape[0]

    # Create solution archive directory
    solutions_dir = tmp_path / "solutions"
    solutions_dir.mkdir()

    # Generate multiple solution files
    rng = np.random.RandomState(42)
    for i in range(100):  # Create enough samples for tests
        solution = rng.randn(n)
        solution_file = solutions_dir / f"solution_{i:03d}.txt"
        np.savetxt(solution_file, solution)

    return str(solutions_dir / "solution_*.txt")


def verify_dataset_consistency(dataset_dir: Path) -> tuple[bool, float]:
    """Verify that A_norm @ x == b_norm for all samples in dataset.

    The zarr pack stores A_norm directly (already in normalized space). RHS and
    solutions are also in normalized space, so no additional scaling is needed.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Tuple of (is_consistent, max_relative_error).
    """
    A = load_matrix_dense_sample(dataset_dir, sample_index=0)
    X_rhs, Y_sol = load_dense_training_arrays(dataset_dir)

    max_error = 0.0
    for i in range(len(X_rhs)):
        b_computed = A @ Y_sol[i]
        b_expected = X_rhs[i]
        residual = np.linalg.norm(b_computed - b_expected)
        norm_b = np.linalg.norm(b_expected)
        relative_error = residual / norm_b if norm_b > 0 else residual
        max_error = max(max_error, relative_error)

    return bool(max_error < 1e-10), float(max_error)


def test_solution_archive_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that solution_archive strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": 10},
                seed=42,
                shuffle=False,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": False,
                    }
                },
            ),
            normalize="matrix",
        ),
        str(tmp_path / "solution-archive"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"solution_archive: A @ x != b, max error = {max_error:.2e}"


def test_eigenvector_forward_consistency(tmp_path: Path, matrix_path: str) -> None:
    """Test that eigenvector_forward strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"eigenvector_forward": 10},
                seed=42,
                shuffle=False,
            ),
            normalize="matrix",
        ),
        str(tmp_path / "eigenvector-forward"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"eigenvector_forward: A @ x != b, max error = {max_error:.2e}"


def test_eigenvector_inverse_consistency(tmp_path: Path, matrix_path: str) -> None:
    """Test that eigenvector_inverse strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"eigenvector_inverse": 10},
                seed=42,
                shuffle=False,
            ),
            normalize="matrix",
        ),
        str(tmp_path / "eigenvector-inverse"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"eigenvector_inverse: A @ x != b, max error = {max_error:.2e}"


def test_no_normalization_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that consistency holds with normalize='none'."""
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": 5},
                seed=42,
                shuffle=False,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": False,
                    }
                },
            ),
            normalize="none",
        ),
        str(tmp_path / "no-normalization"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"normalize='none': A @ x != b, max error = {max_error:.2e}"


def test_mixed_strategies_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that mixing multiple strategies maintains A @ x = b for all samples.

    This verifies that the orchestration correctly handles data from different
    strategies without double-normalization or inconsistencies.
    """
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={
                    "solution_archive": 5,
                    "eigenvector_forward": 5,
                    "eigenvector_inverse": 5,
                },
                seed=42,
                shuffle=True,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": False,
                    }
                },
            ),
            normalize="matrix",
        ),
        str(tmp_path / "mixed-strategies"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"mixed strategies: A @ x != b, max error = {max_error:.2e}"

    # Verify we got the expected total count
    rhs, _ = load_dense_training_arrays(Path(dataset_dir))
    assert len(rhs) == 15, "Expected 15 total samples (5 + 5 + 5)"


def test_mixed_archive_and_synthetic_strategies(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test mixing archive strategies with synthetic strategies.

    This is a key use case: combining real solution archives with synthetic
    data generation strategies.
    """
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={
                    "solution_archive": 10,
                    "eigenvector_forward": 10,
                },
                seed=42,
                shuffle=True,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": False,
                    }
                },
            ),
            normalize="matrix",
        ),
        str(tmp_path / "archive-and-synthetic"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"archive+synthetic: A @ x != b, max error = {max_error:.2e}"

    # Verify total count
    rhs, _ = load_dense_training_arrays(Path(dataset_dir))
    assert len(rhs) == 20, "Expected 20 total samples (10 + 10)"


def test_large_mixed_dataset_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test a larger mixed dataset to ensure consistency at scale."""
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={
                    "solution_archive": 50,
                    "eigenvector_forward": 30,
                    "eigenvector_inverse": 20,
                },
                seed=42,
                shuffle=True,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": True,
                        "seed": 42,
                    }
                },
            ),
            normalize="matrix",
        ),
        str(tmp_path / "large-mixed"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"large mixed dataset: A @ x != b, max error = {max_error:.2e}"

    # Verify total count
    rhs, _ = load_dense_training_arrays(Path(dataset_dir))
    assert len(rhs) == 100, "Expected 100 total samples (50 + 30 + 20)"


def test_mixed_strategies_with_shuffling(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that shuffling mixed strategies doesn't break consistency.

    Shuffling should preserve A @ x = b for each individual sample.
    """
    dataset_dir = build_dataset(
        SourceSpec(
            matrix_path=matrix_path,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={
                    "solution_archive": 8,
                    "eigenvector_forward": 7,
                },
                seed=123,
                shuffle=True,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": solutions_glob,
                        "shuffle": True,
                        "seed": 456,
                    }
                },
            ),
            normalize="matrix",
        ),
        str(tmp_path / "shuffled-mixed"),
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"shuffled mixed: A @ x != b, max error = {max_error:.2e}"
