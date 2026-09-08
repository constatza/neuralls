"""End-to-end tests for per-matrix solution binding via solution_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays


@pytest.fixture
def two_spd_matrices_with_solutions(tmp_path: Path) -> tuple[Path, Path, int]:
    """Two 4x4 SPD matrices and matching solution files paired by integer ID in filename.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Tuple of (mat_dir, sol_dir, n) where mat_dir contains two matrix text
        files, sol_dir contains two solution text files, and n is the matrix size.
    """
    rng = np.random.default_rng(seed=99)
    n = 4
    mat_dir = tmp_path / "matrices"
    sol_dir = tmp_path / "solutions"
    mat_dir.mkdir()
    sol_dir.mkdir()

    for i in range(2):
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        np.savetxt(mat_dir / f"A_{i}.txt", A)
        x = rng.standard_normal(n)
        np.savetxt(sol_dir / f"x_{i}.txt", x)

    return mat_dir, sol_dir, n


def test_solution_binding_loads_per_matrix_solution(
    two_spd_matrices_with_solutions: tuple[Path, Path, int],
    tmp_path: Path,
) -> None:
    """Each matrix binding receives its own pre-loaded solution via solution_path.

    Args:
        two_spd_matrices_with_solutions: Fixture providing (mat_dir, sol_dir, n).
        tmp_path: Pytest-provided temporary directory for dataset output.
    """
    mat_dir, sol_dir, n = two_spd_matrices_with_solutions
    dataset_dir = str(tmp_path / "dataset")

    build_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "A_*.txt"),
            solution_path=str(sol_dir / "x_*.txt"),
            sample_id_regex=r"(\d+)(?!.*\d)",
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": -1},
            ),
        ),
        dataset_dir,
    )

    rhs, solutions = load_dense_training_arrays(dataset_dir)

    # Two matrices x 1 solution each -> 2 samples
    assert rhs.shape == (2, n)
    assert solutions.shape == (2, n)
    assert rhs.dtype == np.float64
    assert solutions.dtype == np.float64
