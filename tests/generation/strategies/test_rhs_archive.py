"""Tests for RhsArchiveStrategy (rhs_archive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.interfaces import SolvedRows
from neuralls.domain.generation.runner import ArchiveStrategyRegistration
from neuralls.domain.generation.strategies.rhs_archive import RhsArchiveStrategy
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import solved_rows

ARCHIVE_DIR = Path("archive")


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def rhs_files(
    archive_source: InMemoryArchiveSource, spd_matrix: np.ndarray
) -> tuple[list[Path], str]:
    """Register five RHS archive files in memory."""
    n = spd_matrix.shape[0]
    rng = np.random.default_rng(3)
    files = [
        archive_source.add(ARCHIVE_DIR / f"rhs_{i:03d}.txt", rng.standard_normal(n))
        for i in range(5)
    ]
    return files, str(ARCHIVE_DIR / "rhs_*.txt")


def test_rhs_archive_registered() -> None:
    """RhsArchiveStrategy is registered under 'rhs_archive'."""
    from neuralls.domain.generation.runner import _registry

    assert "rhs_archive" in _registry._strategies
    registration = _registry._strategies["rhs_archive"]
    assert isinstance(registration, ArchiveStrategyRegistration)
    assert isinstance(registration.strategy, RhsArchiveStrategy)


def test_rhs_archive_shapes(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str], archive_source: InMemoryArchiveSource
) -> None:
    """Loaded RHS and solved solutions have correct shapes."""
    n = spd_matrix.shape[0]
    _files, glob_pattern = rhs_files

    result = solved_rows(
        run_generation(
            "rhs_archive",
            spd_matrix,
            cfg={"rhs_glob": glob_pattern, "samples": 3},
            archive_source=archive_source,
        )
    )
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.shape == (3, n)
    assert result.solutions.shape == (3, n)
    assert isinstance(result, SolvedRows)


def test_rhs_archive_solves_ax_equals_b(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str], archive_source: InMemoryArchiveSource
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

    result = solved_rows(
        run_generation("rhs_archive", spd_matrix, cfg=cfg, archive_source=archive_source)
    )
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.rhs.shape[0]):
        assert result.rhs is not None
        assert result.solutions is not None
        residual = np.linalg.norm(spd_matrix @ result.solutions[i] - result.rhs[i])
        rel = residual / np.linalg.norm(result.rhs[i])
        assert rel < 1e-6, f"Sample {i}: relative residual {rel:.2e}"


def test_rhs_archive_all_files(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str], archive_source: InMemoryArchiveSource
) -> None:
    """samples=-1 loads every available file."""
    files, glob_pattern = rhs_files

    result = solved_rows(
        run_generation(
            "rhs_archive",
            spd_matrix,
            cfg={"rhs_glob": glob_pattern, "samples": -1},
            archive_source=archive_source,
        )
    )
    assert result.rhs is not None

    assert result.rhs.shape[0] == len(files)


def test_rhs_archive_deterministic_shuffle(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str], archive_source: InMemoryArchiveSource
) -> None:
    """Shuffling with the same seed produces identical output on repeated calls."""
    _files, glob_pattern = rhs_files
    cfg = {"rhs_glob": glob_pattern, "samples": 3, "shuffle": True, "seed": 42}

    r1 = solved_rows(
        run_generation("rhs_archive", spd_matrix, cfg=cfg, archive_source=archive_source)
    )
    assert r1.rhs is not None
    r2 = solved_rows(
        run_generation("rhs_archive", spd_matrix, cfg=cfg, archive_source=archive_source)
    )
    assert r2.rhs is not None

    assert r1.rhs.ndim == 2 and r2.rhs.ndim == 2
    np.testing.assert_array_equal(r1.rhs, r2.rhs)


def test_rhs_archive_missing_glob_raises(archive_source: InMemoryArchiveSource) -> None:
    """Missing rhs_glob raises ValidationError (pydantic extra=forbid)."""
    with pytest.raises(ValueError):
        solved_rows(
            run_generation(
                "rhs_archive", np.eye(5), cfg={"samples": 1}, archive_source=archive_source
            )
        )


def test_rhs_archive_insufficient_files_raises(
    spd_matrix: np.ndarray, rhs_files: tuple[list[Path], str], archive_source: InMemoryArchiveSource
) -> None:
    """Requesting more files than available raises ValueError."""
    _files, glob_pattern = rhs_files
    with pytest.raises(ValueError):
        solved_rows(
            run_generation(
                "rhs_archive",
                spd_matrix,
                cfg={"rhs_glob": glob_pattern, "samples": 100},
                archive_source=archive_source,
            )
        )
