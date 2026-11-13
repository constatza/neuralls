"""Tests for ValidatedArchiveStrategy (refactored with SOLID providers + transforms)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.notices import InvalidPairs
from neuralls.domain.generation.strategies.validated_archive import ValidatedArchiveStrategy
from neuralls.domain.generation.strategy_configs import ValidatedArchiveConfig
from tests.fakes.archive_source import InMemoryArchiveSource

MISSING_DIR = Path("archive/missing")
SOLUTIONS_DIR = Path("archive/solutions")
RHS_DIR = Path("archive/rhs")


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def valid_pairs(
    archive_source: InMemoryArchiveSource, sample_matrix: np.ndarray
) -> tuple[Path, Path]:
    """Register valid (solution, RHS) pairs in memory."""
    n = sample_matrix.shape[0]
    rng = np.random.default_rng(11)

    # Generate consistent pairs: b = A @ x
    for i in range(5):
        x = rng.standard_normal(n)
        archive_source.add(SOLUTIONS_DIR / f"sol_{i:03d}.txt", x)
        archive_source.add(RHS_DIR / f"rhs_{i:03d}.txt", sample_matrix @ x)

    return SOLUTIONS_DIR, RHS_DIR


@pytest.fixture
def invalid_pairs(
    archive_source: InMemoryArchiveSource, sample_matrix: np.ndarray
) -> tuple[Path, Path]:
    """Register inconsistent (solution, RHS) pairs in memory."""
    n = sample_matrix.shape[0]
    rng = np.random.default_rng(12)

    # Generate INCONSISTENT pairs: b != A @ x
    for i in range(5):
        archive_source.add(SOLUTIONS_DIR / f"sol_{i:03d}.txt", rng.standard_normal(n))
        archive_source.add(RHS_DIR / f"rhs_{i:03d}.txt", rng.standard_normal(n))

    return SOLUTIONS_DIR, RHS_DIR


def test_validated_archive_strategy_registered() -> None:
    """Test that strategy is properly registered."""
    strategy = ValidatedArchiveStrategy()
    assert strategy.name == "validated_archive"
    assert strategy.ConfigType == ValidatedArchiveConfig


# Note: Tests for _load_paired_vectors and _verify_consistency removed.
# These are now internal to PairedFileInputProvider and VerifyTransform.
# They are tested via test_providers.py and test_transforms.py.


def test_validated_archive_generate_valid(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path], archive_source: InMemoryArchiveSource
) -> None:
    """Test generation with valid pairs."""
    solutions_dir, rhs_dir = valid_pairs

    cfg = {
        "samples": 5,
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()
    samples = strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)
    assert samples.solutions is not None
    assert samples.rhs is not None

    assert samples.matrix is sample_matrix
    assert samples.rhs.shape == (5, 10)
    assert samples.solutions.shape == (5, 10)

    # Verify consistency
    for i in range(5):
        assert samples.solutions is not None
        b_computed = sample_matrix @ samples.solutions[i]
        assert samples.rhs is not None
        residual = np.linalg.norm(b_computed - samples.rhs[i])
        rhs_norm = np.linalg.norm(samples.rhs[i])
        rel_error = residual / rhs_norm
        assert rel_error < 1e-10


def test_validated_archive_generate_invalid_fail(
    sample_matrix: np.ndarray,
    invalid_pairs: tuple[Path, Path],
    archive_source: InMemoryArchiveSource,
) -> None:
    """Test that invalid pairs raise error when fail_on_invalid=True."""
    solutions_dir, rhs_dir = invalid_pairs

    cfg = {
        "samples": 5,
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()

    with pytest.raises(ValueError):
        strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)


def test_validated_archive_generate_invalid_warn(
    sample_matrix: np.ndarray,
    invalid_pairs: tuple[Path, Path],
    archive_source: InMemoryArchiveSource,
) -> None:
    """Invalid pairs are reported and the loaded samples kept when configured to continue."""
    solutions_dir, rhs_dir = invalid_pairs

    cfg = {
        "samples": 5,
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": False,
        "seed": 42,
    }

    samples = ValidatedArchiveStrategy().generate(
        sample_matrix, cfg=cfg, archive_source=archive_source
    )

    assert samples.solutions.shape[0] == 5
    match samples.notices:
        case (InvalidPairs(total=5, tolerance=1e-10),):
            pass
        case other:
            pytest.fail(f"Expected one InvalidPairs notice over 5 pairs, got {other}")


def test_validated_archive_shuffle(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path], archive_source: InMemoryArchiveSource
) -> None:
    """Test that shuffle produces deterministic results with seed."""
    solutions_dir, rhs_dir = valid_pairs

    cfg = {
        "samples": 3,
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "shuffle": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()

    # Run twice with same seed
    samples1 = strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)
    samples2 = strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)

    # Should get same results
    np.testing.assert_array_equal(samples1.rhs, samples2.rhs)
    np.testing.assert_array_equal(samples1.solutions, samples2.solutions)


def test_validated_archive_all_files(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path], archive_source: InMemoryArchiveSource
) -> None:
    """Test loading all files with samples=-1."""
    solutions_dir, rhs_dir = valid_pairs

    cfg = {
        "samples": -1,  # All files
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()
    samples = strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)
    assert samples.rhs is not None

    assert samples.rhs.shape[0] == 5  # All 5 files


def test_validated_archive_missing_glob(
    sample_matrix: np.ndarray, archive_source: InMemoryArchiveSource
) -> None:
    """Test that missing glob pattern raises error."""
    cfg = {
        "samples": 5,
        "solutions_glob": str(MISSING_DIR / "sol_*.txt"),
        "rhs_glob": str(MISSING_DIR / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()

    with pytest.raises(FileNotFoundError):
        strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)


def test_validated_archive_insufficient_files(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path], archive_source: InMemoryArchiveSource
) -> None:
    """Test that requesting more files than available raises error."""
    solutions_dir, rhs_dir = valid_pairs

    cfg = {
        "samples": 100,  # More than available
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()

    with pytest.raises(ValueError):
        strategy.generate(sample_matrix, cfg=cfg, archive_source=archive_source)
