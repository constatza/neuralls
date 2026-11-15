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


@pytest.mark.parametrize("samples", [1, 3])
def test_neutral_ones_solutions_are_ones_and_rhs_is_a_times_ones(
    sample_matrix: np.ndarray, samples: int
) -> None:
    result = NeutralOnesStrategy().generate(
        matrix=sample_matrix, cfg={"samples": samples, "seed": 42}, archive=None
    )

    assert result.solutions is not None
    assert result.rhs is not None
    np.testing.assert_array_equal(result.solutions, np.ones((samples, 3)))
    np.testing.assert_allclose(result.rhs, np.tile(sample_matrix @ np.ones(3), (samples, 1)))


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


def test_neutral_ones_config_rejects_negative_samples() -> None:
    with pytest.raises(ValidationError):
        NeutralOnesConfig(samples=-2, seed=42, shuffle=True)
