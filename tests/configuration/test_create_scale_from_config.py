"""Tests for create_scale_from_config() function."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.normalization import MatrixScale, create_scale_from_config


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Sample symmetric positive-definite matrix."""
    return np.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ]
    )


def test_create_scale_none() -> None:
    """Test 'none' normalization returns None."""
    A = np.eye(3)
    scale = create_scale_from_config("none", A)
    assert scale is None


def test_create_scale_matrix_auto_bound(sample_matrix: np.ndarray) -> None:
    """Test matrix normalization with auto-computed spectral bound."""
    scale = create_scale_from_config("matrix", sample_matrix)
    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound > 0
    assert scale.dimension_scale == np.sqrt(3)


def test_create_scale_matrix_explicit_bound(sample_matrix: np.ndarray) -> None:
    """Test matrix normalization with explicit spectral bound."""
    explicit_bound = 10.0
    scale = create_scale_from_config("matrix", sample_matrix, spectral_radius_bound=explicit_bound)
    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound == explicit_bound
    assert scale.dimension_scale == np.sqrt(3)


def test_create_scale_invalid_type(sample_matrix: np.ndarray) -> None:
    """Test invalid normalization type raises ValueError."""
    with pytest.raises(ValueError):
        create_scale_from_config("invalid", sample_matrix)  # type: ignore[arg-type]
