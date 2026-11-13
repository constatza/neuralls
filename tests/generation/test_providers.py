"""Tests for input providers (Layer 1: Data Provision)."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.interfaces import ArchiveData, ArchiveField
from neuralls.domain.generation.providers import (
    ConstantInputProvider,
    FileInputProvider,
    HybridInputProvider,
    PairedFileInputProvider,
    RandomInputProvider,
    provide_solutions,
)
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.os_paths import FAKE_ROOT

ARCHIVE_DIR = FAKE_ROOT / "archive"


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Sample 10x10 identity matrix."""
    return np.eye(10)


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def sample_rng() -> np.random.Generator:
    """Sample RNG for testing."""
    return np.random.default_rng(42)


def test_random_input_provider_generates_correct_shape(
    sample_matrix: np.ndarray,
    sample_rng: np.random.Generator,
) -> None:
    """Test random provider generates correct shape."""
    provider = RandomInputProvider(seed=42, scale=1.0)
    solutions = provider.provide(matrix=sample_matrix, count=5, rng=sample_rng)

    assert solutions.shape == (5, 10)
    assert solutions.dtype == np.float64


def test_random_input_provider_is_deterministic() -> None:
    """Test random provider is deterministic with seed."""
    matrix = np.eye(10)
    rng = np.random.default_rng(99)  # Different RNG

    provider1 = RandomInputProvider(seed=42, scale=1.0)
    provider2 = RandomInputProvider(seed=42, scale=1.0)

    solutions1 = provider1.provide(matrix, count=5, rng=rng)
    solutions2 = provider2.provide(matrix, count=5, rng=rng)

    assert np.allclose(solutions1, solutions2)


def test_constant_input_provider_generates_ones(
    sample_matrix: np.ndarray,
    sample_rng: np.random.Generator,
) -> None:
    """Test constant provider generates ones."""
    provider = ConstantInputProvider(value=1.0)
    solutions = provider.provide(matrix=sample_matrix, count=5, rng=sample_rng)

    assert solutions.shape == (5, 10)
    assert np.all(solutions == 1.0)


def test_constant_input_provider_custom_value(
    sample_matrix: np.ndarray,
    sample_rng: np.random.Generator,
) -> None:
    """Test constant provider with custom value."""
    provider = ConstantInputProvider(value=3.14)
    solutions = provider.provide(matrix=sample_matrix, count=5, rng=sample_rng)

    assert np.all(solutions == 3.14)


def test_file_input_provider_loads_from_files(archive_source: InMemoryArchiveSource) -> None:
    """Test file provider loads from the archive."""
    rng = np.random.default_rng(1)
    for i in range(10):
        archive_source.add(ARCHIVE_DIR / f"sol_{i}.txt", rng.standard_normal(10))

    provider = FileInputProvider(
        archive_source,
        glob_pattern=str(ARCHIVE_DIR / "sol_*.txt"),
        shuffle=False,
        seed=None,
    )
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    solutions = provider.provide(matrix=matrix, count=5, rng=rng)

    assert solutions.shape == (5, 10)
    assert solutions.dtype == np.float64


def test_file_input_provider_with_shuffling(archive_source: InMemoryArchiveSource) -> None:
    """Test file provider shuffles files deterministically."""
    for i in range(10):
        # Each file has constant value i
        archive_source.add(ARCHIVE_DIR / f"sol_{i}.txt", np.full(10, i, dtype=np.float64))

    provider = FileInputProvider(
        archive_source,
        glob_pattern=str(ARCHIVE_DIR / "sol_*.txt"),
        shuffle=True,
        seed=42,
    )
    matrix = np.eye(10)
    rng = np.random.default_rng(99)

    solutions = provider.provide(matrix=matrix, count=5, rng=rng)

    # Should get different files than [0, 1, 2, 3, 4] due to shuffling
    first_values = solutions[:, 0]
    assert not np.array_equal(first_values, np.array([0, 1, 2, 3, 4]))


def test_hybrid_input_provider_uses_archive_when_available() -> None:
    """Test hybrid provider uses archive when available."""
    archive_data = np.random.default_rng(4).standard_normal((20, 10))
    archive = ArchiveData(lhs=archive_data, rhs=None)
    provider = HybridInputProvider(archive, field=ArchiveField.LHS, scale=1.0)

    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    solutions = provider.provide(matrix=matrix, count=5, rng=rng)

    assert solutions.shape == (5, 10)
    assert np.array_equal(solutions, archive_data[:5])


def test_hybrid_input_provider_fallback_to_random() -> None:
    """Test hybrid provider falls back to random when archive missing."""
    provider = HybridInputProvider(archive=None, field=ArchiveField.LHS, scale=1.0)

    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    solutions = provider.provide(matrix=matrix, count=5, rng=rng)

    assert solutions.shape == (5, 10)
    # Should be random, not zeros
    assert not np.all(solutions == 0)


def test_hybrid_input_provider_raises_on_insufficient_archive() -> None:
    """Test hybrid provider raises when archive has too few vectors."""
    archive_data = np.random.default_rng(5).standard_normal((3, 10))  # Only 3 vectors
    archive = ArchiveData(lhs=archive_data, rhs=None)
    provider = HybridInputProvider(archive, field=ArchiveField.LHS, scale=1.0)

    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        provider.provide(matrix=matrix, count=5, rng=rng)


def test_paired_file_input_provider_loads_pairs(archive_source: InMemoryArchiveSource) -> None:
    """Test paired file provider loads solution-RHS pairs."""
    rng = np.random.default_rng(2)
    for i in range(5):
        archive_source.add(ARCHIVE_DIR / f"sol_{i}.txt", rng.standard_normal(10))
        archive_source.add(ARCHIVE_DIR / f"rhs_{i}.txt", rng.standard_normal(10))

    provider = PairedFileInputProvider(
        archive_source,
        solution_glob=str(ARCHIVE_DIR / "sol_*.txt"),
        rhs_glob=str(ARCHIVE_DIR / "rhs_*.txt"),
        shuffle=False,
        seed=None,
    )
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    solutions, rhs = provider.provide(matrix=matrix, count=5, rng=rng)

    assert solutions.shape == (5, 10)
    assert rhs.shape == (5, 10)
    assert solutions.dtype == np.float64
    assert rhs.dtype == np.float64


def test_provide_solutions_prefers_archive_over_glob(
    archive_source: InMemoryArchiveSource,
) -> None:
    """When both an archive and a solutions_glob are available, archive wins."""
    for i in range(5):
        archive_source.add(ARCHIVE_DIR / f"sol_{i}.txt", np.full(10, -1.0, dtype=np.float64))

    archive_data = np.random.default_rng(3).standard_normal((5, 10))
    archive = ArchiveData(lhs=archive_data, rhs=None)
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    result = provide_solutions(
        matrix,
        3,
        rng,
        archive_source=archive_source,
        solutions_glob=str(ARCHIVE_DIR / "sol_*.txt"),
        archive=archive,
        shuffle=False,
        seed=None,
        strategy_name="residuals",
    )

    assert np.array_equal(result, archive_data[:3])


def test_file_input_provider_selects_the_given_file_indices(
    archive_source: InMemoryArchiveSource,
) -> None:
    """Bindings with the same glob and count but different explicit files load different vectors."""
    # File k holds the constant vector k + 1, so its position in the sorted pool is recoverable.
    for idx in range(5):
        archive_source.add(ARCHIVE_DIR / f"solution_{idx:03d}.txt", np.full(4, float(idx + 1)))
    glob_pattern = str(ARCHIVE_DIR / "solution_*.txt")
    matrix = np.eye(4)
    rng = np.random.default_rng(0)

    first = FileInputProvider(
        archive_source, glob_pattern=glob_pattern, file_indices=(0, 1)
    ).provide(matrix=matrix, count=2, rng=rng)
    second = FileInputProvider(
        archive_source, glob_pattern=glob_pattern, file_indices=(3, 4)
    ).provide(matrix=matrix, count=2, rng=rng)

    np.testing.assert_array_equal(first[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(second[:, 0], [4.0, 5.0])
