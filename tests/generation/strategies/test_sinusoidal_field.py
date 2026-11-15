"""Tests for SinusoidalFieldStrategy (sinusoidal_field)."""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.semantics import classify_strategy_row_kind
from neuralls.shared.types import GenerationStrategyKind, RowKind
from tests.generation_support import solved_rows


def test_sinusoidal_field_matches_closed_form(spd_matrix: np.ndarray) -> None:
    """entry i of each row is sin(omega/n * i + phi) for that row's own phase."""
    n = spd_matrix.shape[0]
    omega = 4.0
    cfg = {"samples": 3, "seed": 0, "omega": omega}

    result = solved_rows(run_generation("sinusoidal_field", spd_matrix, cfg=cfg))
    assert result.rhs is not None
    i = np.arange(n)
    for row in result.rhs:
        # Recover the per-row phase from entry 0 and check the rest of the row
        # against the closed form sin(omega/n * i + phi).
        phi = np.arcsin(np.clip(row[0], -1.0, 1.0))
        expected_a = np.sin((omega / n) * i + phi)
        expected_b = np.sin((omega / n) * i + (np.pi - phi))
        assert np.allclose(row, expected_a, atol=1e-8) or np.allclose(row, expected_b, atol=1e-8)


def test_sinusoidal_field_row_kind_is_standard() -> None:
    """SINUSOIDAL_FIELD classifies as RowKind.STANDARD, not CG_INTERNAL."""
    assert classify_strategy_row_kind(GenerationStrategyKind.SINUSOIDAL_FIELD) == RowKind.STANDARD
