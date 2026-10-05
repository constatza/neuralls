"""Tests for input providers (Layer 1: Data Provision)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.domain.generation.interfaces import ArchiveData, ArchiveField
from neuralls.domain.generation.providers import (
    ConstantInputProvider,
    FileInputProvider,
    HybridInputProvider,
    PairedFileInputProvider,
    RandomInputProvider,
    _load_archive_files_cached,
    provide_solutions,
)


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Sample 10x10 identity matrix."""
    return np.eye(10)


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


def test_file_input_provider_loads_from_files(tmp_path) -> None:
    """Test file provider loads from disk."""
    # Create test files
    for i in range(10):
        data = np.random.randn(10)
        np.savetxt(tmp_path / f"sol_{i}.txt", data)

    provider = FileInputProvider(
        glob_pattern=str(tmp_path / "sol_*.txt"),
        shuffle=False,
        seed=None,
    )
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    solutions = provider.provide(matrix=matrix, count=5, rng=rng)

    assert solutions.shape == (5, 10)
    assert solutions.dtype == np.float64


def test_file_input_provider_caches_repeated_identical_calls(tmp_path) -> None:
    """Two calls with identical (glob, count, shuffle, seed, skip) hit disk only once."""
    _load_archive_files_cached.cache_clear()
    for i in range(5):
        np.savetxt(tmp_path / f"sol_{i}.txt", np.full(10, i, dtype=np.float64))

    provider = FileInputProvider(glob_pattern=str(tmp_path / "sol_*.txt"), shuffle=False, seed=None)
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    first = provider.provide(matrix=matrix, count=3, rng=rng)

    # Mutate a file on disk; a cached second call must not see the change.
    np.savetxt(tmp_path / "sol_0.txt", np.full(10, 999.0, dtype=np.float64))
    second = provider.provide(matrix=matrix, count=3, rng=rng)

    assert np.array_equal(first, second)
    assert _load_archive_files_cached.cache_info().hits == 1


def test_file_input_provider_with_shuffling(tmp_path) -> None:
    """Test file provider shuffles files deterministically."""
    # Create test files
    for i in range(10):
        data = np.full(10, i, dtype=np.float64)  # Each file has constant value i
        np.savetxt(tmp_path / f"sol_{i}.txt", data)

    provider = FileInputProvider(
        glob_pattern=str(tmp_path / "sol_*.txt"),
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
    archive_data = np.random.randn(20, 10)
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
    archive_data = np.random.randn(3, 10)  # Only 3 vectors
    archive = ArchiveData(lhs=archive_data, rhs=None)
    provider = HybridInputProvider(archive, field=ArchiveField.LHS, scale=1.0)

    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="Not enough archive"):
        provider.provide(matrix=matrix, count=5, rng=rng)


def test_paired_file_input_provider_loads_pairs(tmp_path) -> None:
    """Test paired file provider loads solution-RHS pairs."""
    # Create test files
    for i in range(5):
        sol = np.random.randn(10)
        rhs = np.random.randn(10)
        np.savetxt(tmp_path / f"sol_{i}.txt", sol)
        np.savetxt(tmp_path / f"rhs_{i}.txt", rhs)

    provider = PairedFileInputProvider(
        solution_glob=str(tmp_path / "sol_*.txt"),
        rhs_glob=str(tmp_path / "rhs_*.txt"),
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


def test_provide_solutions_prefers_archive_over_glob(tmp_path) -> None:
    """When both an archive and a solutions_glob are available, archive wins."""
    for i in range(5):
        np.savetxt(tmp_path / f"sol_{i}.txt", np.full(10, -1.0, dtype=np.float64))

    archive_data = np.random.randn(5, 10)
    archive = ArchiveData(lhs=archive_data, rhs=None)
    matrix = np.eye(10)
    rng = np.random.default_rng(42)

    result = provide_solutions(
        matrix,
        3,
        rng,
        solutions_glob=str(tmp_path / "sol_*.txt"),
        archive=archive,
        shuffle=False,
        seed=None,
        strategy_name="residuals",
    )

    assert np.array_equal(result, archive_data[:3])


def test_file_input_provider_cache_key_includes_file_indices(
    write_solution_files: Callable[[int], str],
) -> None:
    """Bindings with the same glob and count but different explicit files must not share a cache entry."""
    _load_archive_files_cached.cache_clear()
    glob_pattern = write_solution_files(5)
    matrix = np.eye(4)
    rng = np.random.default_rng(0)

    first = FileInputProvider(glob_pattern=glob_pattern, file_indices=(0, 1)).provide(
        matrix=matrix, count=2, rng=rng
    )
    second = FileInputProvider(glob_pattern=glob_pattern, file_indices=(3, 4)).provide(
        matrix=matrix, count=2, rng=rng
    )

    np.testing.assert_array_equal(first[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(second[:, 0], [4.0, 5.0])
