"""Tests for RhsArchiveStrategy (rhs_archive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.runner import MatrixStrategyRegistration
from neuralls.domain.generation.strategies.rhs_archive import RhsArchiveStrategy


@pytest.fixture
def rhs_files(tmp_path: Path, spd_matrix: np.ndarray) -> tuple[list[Path], str]:
    """Create temporary RHS archive files."""
    n = spd_matrix.shape[0]
    rng = np.random.default_rng(3)
    files = []
    for i in range(5):
        p = tmp_path / f"rhs_{i:03d}.txt"
        np.savetxt(p, rng.standard_normal(n))
        files.append(p)
    return files, str(tmp_path / "rhs_*.txt")


def test_rhs_archive_registered() -> None:
    """RhsArchiveStrategy is registered under 'rhs_archive'."""
    from neuralls.domain.generation.runner import _registry

    assert "rhs_archive" in _registry._strategies
    registration = _registry._strategies["rhs_archive"]
    assert isinstance(registration, MatrixStrategyRegistration)
    assert isinstance(registration.strategy, RhsArchiveStrategy)


def test_rhs_archive_shapes(spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str]) -> None:
    """Loaded RHS and solved solutions have correct shapes."""
    n = spd_matrix.shape[0]
    _files, glob_pattern = rhs_files

    result = run_generation("rhs_archive", spd_matrix, cfg={"rhs_glob": glob_pattern, "samples": 3})
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.shape == (3, n)
    assert result.solutions.shape == (3, n)
    assert result.residual_traces is None
    assert result.error_traces is None


def test_rhs_archive_solves_ax_equals_b(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str]
) -> None:
    """Returned solutions satisfy A @ x = b within tolerance."""
    _files, glob_pattern = rhs_files
    cfg = {
        "rhs_glob": glob_pattern,
        "samples": 2,
        "solve_config": {
            "method": "cg",
            "rtol": 1e-10,
            "max_iters": 100,
        },
    }

    result = run_generation("rhs_archive", spd_matrix, cfg=cfg)
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.rhs.shape[0]):
        assert result.rhs is not None
        assert result.solutions is not None
        residual = np.linalg.norm(spd_matrix @ result.solutions[i] - result.rhs[i])
        rel = residual / np.linalg.norm(result.rhs[i])
        assert rel < 1e-6, f"Sample {i}: relative residual {rel:.2e}"


def test_rhs_archive_all_files(spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str]) -> None:
    """samples=-1 loads every available file."""
    files, glob_pattern = rhs_files

    result = run_generation(
        "rhs_archive", spd_matrix, cfg={"rhs_glob": glob_pattern, "samples": -1}
    )
    assert result.rhs is not None

    assert result.rhs.shape[0] == len(files)


def test_rhs_archive_deterministic_shuffle(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str]
) -> None:
    """Shuffling with the same seed produces identical output on repeated calls."""
    _files, glob_pattern = rhs_files
    cfg = {"rhs_glob": glob_pattern, "samples": 3, "shuffle": True, "seed": 42}

    r1 = run_generation("rhs_archive", spd_matrix, cfg=cfg)
    assert r1.rhs is not None
    r2 = run_generation("rhs_archive", spd_matrix, cfg=cfg)
    assert r2.rhs is not None

    assert r1.rhs.ndim == 2 and r2.rhs.ndim == 2
    np.testing.assert_array_equal(r1.rhs, r2.rhs)


def test_rhs_archive_missing_glob_raises() -> None:
    """Missing rhs_glob raises ValidationError (pydantic extra=forbid)."""
    with pytest.raises(ValueError, match="rhs_glob"):
        run_generation("rhs_archive", np.eye(5), cfg={"samples": 1})


def test_rhs_archive_insufficient_files_raises(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str]
) -> None:
    """Requesting more files than available raises ValueError."""
    _files, glob_pattern = rhs_files
    with pytest.raises(ValueError, match="only.*available"):
        run_generation("rhs_archive", spd_matrix, cfg={"rhs_glob": glob_pattern, "samples": 100})
