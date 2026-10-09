"""Tests for SinusoidalFieldStrategy (sinusoidal_field)."""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.semantics import classify_strategy_row_kind
from neuralls.shared.types import GenerationStrategyKind, RowKind


def test_sinusoidal_field_registered() -> None:
    """SinusoidalFieldStrategy is registered under 'sinusoidal_field'."""
    from neuralls.domain.generation.runner import _registry

    assert "sinusoidal_field" in _registry._strategies


def test_sinusoidal_field_shapes(spd_matrix: np.ndarray) -> None:
    """Output arrays have correct shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0, "omega": 1.0}

    result = run_generation("sinusoidal_field", spd_matrix, cfg=cfg)
    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape == (4, n)
    assert result.solutions.shape == (4, n)


def test_sinusoidal_field_matches_closed_form(spd_matrix: np.ndarray) -> None:
    """entry i of each row is sin(omega/n * i + phi) for that row's own phase."""
    n = spd_matrix.shape[0]
    omega = 4.0
    cfg = {"samples": 3, "seed": 0, "omega": omega}

    result = run_generation("sinusoidal_field", spd_matrix, cfg=cfg)
    assert result.rhs is not None
    i = np.arange(n)
    for row in result.rhs:
        # Recover the per-row phase from entry 0 and check the rest of the row
        # against the closed form sin(omega/n * i + phi).
        phi = np.arcsin(np.clip(row[0], -1.0, 1.0))
        expected_a = np.sin((omega / n) * i + phi)
        expected_b = np.sin((omega / n) * i + (np.pi - phi))
        assert np.allclose(row, expected_a, atol=1e-8) or np.allclose(row, expected_b, atol=1e-8)


def test_sinusoidal_field_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed produces identical output across two calls."""
    cfg = {"samples": 4, "seed": 7, "omega": 1.0}

    result1 = run_generation("sinusoidal_field", spd_matrix, cfg=cfg)
    result2 = run_generation("sinusoidal_field", spd_matrix, cfg=cfg)
    assert result1.rhs is not None
    assert result2.rhs is not None
    np.testing.assert_array_equal(result1.rhs, result2.rhs)


def test_sinusoidal_field_different_seeds(spd_matrix: np.ndarray) -> None:
    """Different seeds produce different output."""
    cfg_a = {"samples": 4, "seed": 1, "omega": 1.0}
    cfg_b = {"samples": 4, "seed": 2, "omega": 1.0}

    result1 = run_generation("sinusoidal_field", spd_matrix, cfg=cfg_a)
    result2 = run_generation("sinusoidal_field", spd_matrix, cfg=cfg_b)
    assert result1.rhs is not None
    assert result2.rhs is not None
    assert not np.array_equal(result1.rhs, result2.rhs)


def test_sinusoidal_field_row_kind_is_standard() -> None:
    """SINUSOIDAL_FIELD classifies as RowKind.STANDARD, not CG_INTERNAL."""
    assert classify_strategy_row_kind(GenerationStrategyKind.SINUSOIDAL_FIELD) == RowKind.STANDARD
