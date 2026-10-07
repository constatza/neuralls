"""Tests for SolutionArchiveStrategy (solution_archive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.runner import MatrixStrategyRegistration
from neuralls.domain.generation.strategies.solution_archive import SolutionArchiveStrategy


@pytest.fixture
def solution_files(tmp_path: Path, spd_matrix: np.ndarray) -> tuple[list[Path], str]:
    """Create temporary solution archive files."""
    n = spd_matrix.shape[0]
    rng = np.random.default_rng(5)
    files = []
    for i in range(5):
        p = tmp_path / f"sol_{i:03d}.txt"
        np.savetxt(p, rng.standard_normal(n))
        files.append(p)
    return files, str(tmp_path / "sol_*.txt")


def test_solution_archive_registered() -> None:
    """SolutionArchiveStrategy is registered under 'solution_archive'."""
    from neuralls.domain.generation.runner import _registry

    assert "solution_archive" in _registry._strategies
    registration = _registry._strategies["solution_archive"]
    assert isinstance(registration, MatrixStrategyRegistration)
    assert isinstance(registration.strategy, SolutionArchiveStrategy)


def test_solution_archive_shapes(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """Loaded solutions and computed RHS have correct shapes."""
    n = spd_matrix.shape[0]
    _files, glob_pattern = solution_files

    result = run_generation(
        "solution_archive", spd_matrix, cfg={"solutions_glob": glob_pattern, "samples": 3}
    )
    assert result.rhs is not None
    assert result.solutions is not None

    assert result.solutions.shape == (3, n)
    assert result.rhs.shape == (3, n)
    assert result.residual_traces is None
    assert result.error_traces is None


def test_solution_archive_rhs_equals_ax(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """RHS = A @ x for every loaded solution."""
    _files, glob_pattern = solution_files

    result = run_generation(
        "solution_archive", spd_matrix, cfg={"solutions_glob": glob_pattern, "samples": 3}
    )
    assert result.rhs is not None
    assert result.solutions is not None

    assert result.solutions.ndim == 2 and result.rhs.ndim == 2
    for i in range(result.solutions.shape[0]):
        assert result.rhs is not None
        assert result.solutions is not None
        np.testing.assert_allclose(result.rhs[i], spd_matrix @ result.solutions[i], rtol=1e-12)


def test_solution_archive_all_files(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """samples=-1 loads every available file."""
    files, glob_pattern = solution_files

    result = run_generation(
        "solution_archive", spd_matrix, cfg={"solutions_glob": glob_pattern, "samples": -1}
    )
    assert result.solutions is not None

    assert result.solutions.shape[0] == len(files)


def test_solution_archive_deterministic_shuffle(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """Shuffling with the same seed produces identical output on repeated calls."""
    _files, glob_pattern = solution_files
    cfg = {"solutions_glob": glob_pattern, "samples": 3, "shuffle": True, "seed": 42}

    r1 = run_generation("solution_archive", spd_matrix, cfg=cfg)
    assert r1.solutions is not None
    r2 = run_generation("solution_archive", spd_matrix, cfg=cfg)
    assert r2.solutions is not None

    assert r1.solutions.ndim == 2 and r2.solutions.ndim == 2
    np.testing.assert_array_equal(r1.solutions, r2.solutions)


def test_solution_archive_skip_selects_later_files(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """skip offsets deterministic archive selection before loading vectors."""
    files, glob_pattern = solution_files

    result = run_generation(
        "solution_archive",
        spd_matrix,
        cfg={"solutions_glob": glob_pattern, "samples": 2, "shuffle": False, "skip": 2},
    )

    assert result.solutions is not None
    expected = np.array([np.loadtxt(files[2]), np.loadtxt(files[3])], dtype=np.float64)
    np.testing.assert_array_equal(result.solutions, expected)


def test_solution_archive_skip_applies_after_shuffle(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """Shuffled archives skip within the deterministic permutation."""
    files, glob_pattern = solution_files
    seed = 42
    skip = 1
    count = 2

    result = run_generation(
        "solution_archive",
        spd_matrix,
        cfg={
            "solutions_glob": glob_pattern,
            "samples": count,
            "shuffle": True,
            "seed": seed,
            "skip": skip,
        },
    )

    assert result.solutions is not None
    indices = np.random.default_rng(seed).permutation(len(files))[skip : skip + count]
    expected = np.array([np.loadtxt(files[idx]) for idx in indices], dtype=np.float64)
    np.testing.assert_array_equal(result.solutions, expected)


def test_solution_archive_skip_plus_samples_beyond_available_raises(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """skip and samples are validated together against available files."""
    _, glob_pattern = solution_files
    with pytest.raises(ValueError, match="skip=4.*only 5 available"):
        run_generation(
            "solution_archive",
            spd_matrix,
            cfg={"solutions_glob": glob_pattern, "samples": 2, "skip": 4},
        )


def test_solution_archive_missing_glob_raises() -> None:
    """Missing solutions_glob raises ValidationError."""
    with pytest.raises(ValueError, match="solutions_glob"):
        run_generation("solution_archive", np.eye(5), cfg={"samples": 1})


def test_solution_archive_insufficient_files_raises(
    spd_matrix: np.ndarray, solution_files: tuple[list[Path], str]
) -> None:
    """Requesting more files than available raises ValueError."""
    _files, glob_pattern = solution_files
    with pytest.raises(ValueError, match="only.*available"):
        run_generation(
            "solution_archive", spd_matrix, cfg={"solutions_glob": glob_pattern, "samples": 100}
        )
