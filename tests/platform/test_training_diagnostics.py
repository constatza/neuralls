"""Tests for `compute_diagnostics`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuralls.platform.reporting.training_diagnostics import compute_diagnostics


@pytest.fixture
def targets() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(50, 1))


def test_perfect_prediction_has_unit_r2_and_scale_ratio(targets: np.ndarray) -> None:
    metrics = compute_diagnostics(targets, targets).metrics

    assert metrics["eval/rmse"] == 0.0
    assert metrics["eval/r2"] == pytest.approx(1.0)
    assert metrics["eval/pred_target_std_ratio"] == pytest.approx(1.0)
    assert metrics["eval/mean_shift"] == pytest.approx(0.0)


def test_scaled_and_shifted_prediction_exposes_normalization_mismatch(
    targets: np.ndarray,
) -> None:
    metrics = compute_diagnostics(2.0 * targets + 1.0, targets).metrics

    assert metrics["eval/pred_target_std_ratio"] == pytest.approx(2.0)
    assert metrics["eval/mean_shift"] == pytest.approx(
        (1.0 + targets.mean()) / targets.std(), rel=1e-9
    )


def test_constant_targets_give_nan_scale_metrics() -> None:
    metrics = compute_diagnostics(np.ones(5), np.zeros(5)).metrics

    assert all(
        math.isnan(metrics[key])
        for key in ("eval/r2", "eval/pred_target_std_ratio", "eval/mean_shift")
    )
