"""Tests for ValidatedArchiveStrategy (refactored with SOLID providers + transforms)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.domain.generation.strategies.validated_archive import ValidatedArchiveStrategy
from neuralls.domain.generation.strategy_configs import ValidatedArchiveConfig


@pytest.fixture
def valid_pairs(tmp_path: Path, sample_matrix: np.ndarray) -> tuple[Path, Path]:
    """Create valid (solution, RHS) pairs on disk."""
    n = sample_matrix.shape[0]
    num_samples = 5

    solutions_dir = tmp_path / "solutions"
    rhs_dir = tmp_path / "rhs"
    solutions_dir.mkdir()
    rhs_dir.mkdir()

    # Generate consistent pairs: b = A @ x
    for i in range(num_samples):
        x = np.random.randn(n)
        b = sample_matrix @ x

        np.savetxt(solutions_dir / f"sol_{i:03d}.txt", x)
        np.savetxt(rhs_dir / f"rhs_{i:03d}.txt", b)

    return solutions_dir, rhs_dir


@pytest.fixture
def invalid_pairs(tmp_path: Path, sample_matrix: np.ndarray) -> tuple[Path, Path]:
    """Create invalid (solution, RHS) pairs on disk."""
    n = sample_matrix.shape[0]
    num_samples = 5

    solutions_dir = tmp_path / "solutions"
    rhs_dir = tmp_path / "rhs"
    solutions_dir.mkdir()
    rhs_dir.mkdir()

    # Generate INCONSISTENT pairs: b != A @ x
    for i in range(num_samples):
        x = np.random.randn(n)
        b = np.random.randn(n)  # Random, not A @ x!

        np.savetxt(solutions_dir / f"sol_{i:03d}.txt", x)
        np.savetxt(rhs_dir / f"rhs_{i:03d}.txt", b)

    return solutions_dir, rhs_dir


def test_validated_archive_strategy_registered() -> None:
    """Test that strategy is properly registered."""
    strategy = ValidatedArchiveStrategy()
    assert strategy.name == "validated_archive"
    assert strategy.ConfigType == ValidatedArchiveConfig


# Note: Tests for _load_paired_vectors and _verify_consistency removed.
# These are now internal to PairedFileInputProvider and VerifyTransform.
# They are tested via test_providers.py and test_transforms.py.


def test_validated_archive_generate_valid(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path]
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
    samples = strategy.generate(sample_matrix, cfg=cfg)
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
    sample_matrix: np.ndarray, invalid_pairs: tuple[Path, Path]
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

    with pytest.raises(ValueError, match="Found .* invalid pairs"):
        strategy.generate(sample_matrix, cfg=cfg)


def test_validated_archive_generate_invalid_warn(
    sample_matrix: np.ndarray, invalid_pairs: tuple[Path, Path]
) -> None:
    """Invalid pairs warn and still return loaded samples when configured to continue."""
    solutions_dir, rhs_dir = invalid_pairs

    cfg = {
        "samples": 5,
        "solutions_glob": str(solutions_dir / "sol_*.txt"),
        "rhs_glob": str(rhs_dir / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": False,  # Just warn
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()
    with patch(
        "neuralls.domain.generation.strategies.validated_archive.logger.warning"
    ) as mock_warning:
        samples = strategy.generate(sample_matrix, cfg=cfg)

    assert samples.solutions is not None
    mock_warning.assert_called_once()


def test_validated_archive_shuffle(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path]
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
    samples1 = strategy.generate(sample_matrix, cfg=cfg)
    samples2 = strategy.generate(sample_matrix, cfg=cfg)

    # Should get same results
    np.testing.assert_array_equal(samples1.rhs, samples2.rhs)
    np.testing.assert_array_equal(samples1.solutions, samples2.solutions)


def test_validated_archive_all_files(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path]
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
    samples = strategy.generate(sample_matrix, cfg=cfg)
    assert samples.rhs is not None

    assert samples.rhs.shape[0] == 5  # All 5 files


def test_validated_archive_missing_glob(
    sample_matrix: np.ndarray,
    tmp_path: Path,
) -> None:
    """Test that missing glob pattern raises error."""
    cfg = {
        "samples": 5,
        "solutions_glob": str(tmp_path / "missing" / "sol_*.txt"),
        "rhs_glob": str(tmp_path / "missing" / "rhs_*.txt"),
        "verification_tolerance": 1e-10,
        "fail_on_invalid": True,
        "seed": 42,
    }

    strategy = ValidatedArchiveStrategy()

    with pytest.raises(FileNotFoundError):
        strategy.generate(sample_matrix, cfg=cfg)


def test_validated_archive_insufficient_files(
    sample_matrix: np.ndarray, valid_pairs: tuple[Path, Path]
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

    with pytest.raises(ValueError, match="Requested .* files but only .* available"):
        strategy.generate(sample_matrix, cfg=cfg)
