"""Characterization tests for deriving a mother RHS from a solution archive.

``provide_rhs = true`` with no explicit ``rhs_path`` makes the generate
workflow synthesize one: it picks the lexicographically first solution file
matching the configured glob, computes ``b = A @ x`` from it, and writes that
vector to ``mother-rhs.txt`` in the dataset directory.

These tests pin the computation and the persistence separately — which
solution file is chosen and what vector comes out of it, versus where the
file lands and exactly how it is encoded on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation._archive_resolution import (
    compute_rhs_from_solution,
    persist_derived_rhs,
)


@pytest.fixture
def matrix() -> np.ndarray:
    """A small non-symmetric matrix, so ``A @ x`` differs from ``x @ A``."""
    return np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]], dtype=np.float64)


@pytest.fixture
def solution_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Two distinct solution vectors; only the first must ever be used."""
    return (
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([9.0, 9.0, 9.0], dtype=np.float64),
    )


@pytest.fixture
def solutions_dir(tmp_path: Path, solution_vectors: tuple[np.ndarray, np.ndarray]) -> Path:
    """Directory of solution files named so sorting order is unambiguous."""
    directory = tmp_path / "solutions"
    directory.mkdir()
    first, second = solution_vectors
    np.savetxt(directory / "sol_001.txt", first)
    np.savetxt(directory / "sol_002.txt", second)
    return directory


@pytest.fixture
def solutions_glob(solutions_dir: Path) -> str:
    """Glob matching every solution file in ``solutions_dir``."""
    return str(solutions_dir / "sol_*.txt")


def test_computes_a_at_x_for_the_first_matching_solution(
    matrix: np.ndarray,
    solutions_glob: str,
    solution_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    """The lexicographically first file wins, and the result is exactly ``A @ x``."""
    first, _ = solution_vectors

    rhs = compute_rhs_from_solution(matrix=matrix, solutions_glob=solutions_glob)

    np.testing.assert_allclose(rhs, matrix @ first)
    np.testing.assert_allclose(rhs, np.array([4.0, 9.0, 13.0]))


def test_computation_writes_nothing(
    tmp_path: Path, matrix: np.ndarray, solutions_glob: str, solutions_dir: Path
) -> None:
    """The compute step is read-only — persistence is a separate, explicit call."""
    before = sorted(p.name for p in tmp_path.iterdir())

    compute_rhs_from_solution(matrix=matrix, solutions_glob=solutions_glob)

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / "mother-rhs.txt").exists()


def test_missing_solutions_directory_raises(tmp_path: Path, matrix: np.ndarray) -> None:
    with pytest.raises(FileNotFoundError, match="Solutions directory not found"):
        compute_rhs_from_solution(
            matrix=matrix, solutions_glob=str(tmp_path / "absent" / "sol_*.txt")
        )


def test_directory_without_matching_files_raises(
    tmp_path: Path, matrix: np.ndarray, solutions_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="No solution files available"):
        compute_rhs_from_solution(matrix=matrix, solutions_glob=str(solutions_dir / "none_*.txt"))


def test_persists_to_mother_rhs_txt_in_the_dataset_dir(tmp_path: Path) -> None:
    rhs = np.array([4.0, 9.0, 13.0], dtype=np.float64)
    dataset_dir = tmp_path / "dataset"

    written = persist_derived_rhs(rhs, dataset_dir=dataset_dir)

    assert written == dataset_dir / "mother-rhs.txt"
    np.testing.assert_allclose(np.loadtxt(written), rhs)


def test_persistence_creates_missing_parent_directories(tmp_path: Path) -> None:
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    dataset_dir = tmp_path / "deeply" / "nested" / "dataset"

    written = persist_derived_rhs(rhs, dataset_dir=dataset_dir)

    assert written.exists()
    assert dataset_dir.is_dir()


def test_persisted_encoding_is_full_precision_scientific(tmp_path: Path) -> None:
    """The ``%.18e`` format is what downstream txt readers expect — pin it exactly."""
    rhs = np.array([1.0 / 3.0, 2.0], dtype=np.float64)

    written = persist_derived_rhs(rhs, dataset_dir=tmp_path / "dataset")

    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines == ["3.333333333333333148e-01", "2.000000000000000000e+00"]
