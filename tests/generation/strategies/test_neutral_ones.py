"""Tests for neutral_ones strategy."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation.strategies.neutral_ones import (
    NeutralOnesConfig,
    NeutralOnesStrategy,
)


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Create a simple 3x3 test matrix."""
    return np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float64)


def test_neutral_ones_strategy_basic(sample_matrix: np.ndarray) -> None:
    """Test neutral_ones strategy generates x=ones and b=A@x."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 1, "seed": 42}

    result = strategy.generate(
        matrix=sample_matrix,
        cfg=config,
        archive=None,
    )
    assert result.solutions is not None
    assert result.rhs is not None

    # Check shapes
    assert result.rhs.shape == (1, 3)
    assert result.solutions.shape == (1, 3)

    # Check solution is all ones
    np.testing.assert_array_equal(result.solutions, np.ones((1, 3)))

    # Check b = A @ x
    expected_rhs = sample_matrix @ np.ones(3)
    assert result.rhs is not None
    np.testing.assert_array_almost_equal(result.rhs[0], expected_rhs)


def test_neutral_ones_multiple_samples(sample_matrix: np.ndarray) -> None:
    """Test neutral_ones strategy with multiple samples."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 3, "seed": 42}

    result = strategy.generate(
        matrix=sample_matrix,
        cfg=config,
        archive=None,
    )
    assert result.solutions is not None
    assert result.rhs is not None

    # Check shapes
    assert result.rhs.shape == (3, 3)
    assert result.solutions.shape == (3, 3)

    # All solutions should be ones
    np.testing.assert_array_equal(result.solutions, np.ones((3, 3)))

    # All RHS should be identical (A @ ones)
    expected_rhs = sample_matrix @ np.ones(3)
    for i in range(3):
        np.testing.assert_array_almost_equal(result.rhs[i], expected_rhs)


def test_neutral_ones_deterministic(sample_matrix: np.ndarray) -> None:
    """Test that neutral_ones produces deterministic results."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 2, "seed": 42}

    result1 = strategy.generate(matrix=sample_matrix, cfg=config, archive=None)
    assert result1.solutions is not None
    assert result1.rhs is not None
    result2 = strategy.generate(matrix=sample_matrix, cfg=config, archive=None)
    assert result2.solutions is not None
    assert result2.rhs is not None

    assert result1.rhs.ndim == 2 and result2.rhs.ndim == 2
    assert result1.solutions.ndim == 2 and result2.solutions.ndim == 2
    np.testing.assert_array_equal(result1.rhs, result2.rhs)
    np.testing.assert_array_equal(result1.solutions, result2.solutions)


def test_neutral_ones_config_validation() -> None:
    """Test NeutralOnesConfig validation."""
    # Valid config
    config = NeutralOnesConfig(samples=1, seed=42, shuffle=True)
    assert config.samples == 1
    assert config.seed == 42

    # Negative samples should raise validation error
    with pytest.raises(ValidationError):
        NeutralOnesConfig(samples=-2, seed=42, shuffle=True)
