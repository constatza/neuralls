"""Tests for the config-to-torchalg weighting bridge (`composition/preconditioners/_weighting.py`).

The actual weighting math lives in `torchalg.preconditioners.implementations.pod.weighting`
(tested there); this module's only job is dispatching a `SnapshotWeightingConfig`
to the right torchalg function, so these tests only need to confirm that
dispatch — not re-verify the math.
"""

from __future__ import annotations

import pytest
import torch
from torchalg.preconditioners.implementations.pod import (
    power_norm_scales,
    smoother_persistence_scales,
)

from neuralls.composition.preconditioners._weighting import (
    resolve_row_scales,
    weighting_needs_matrix,
)
from neuralls.platform.config.models.preconditioner import (
    PowerNormWeightingConfig,
    RawWeightingConfig,
    SmootherPersistenceWeightingConfig,
)


@pytest.fixture
def snapshots() -> torch.Tensor:
    """Small snapshot ensemble for bridge dispatch tests."""
    return torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [3.0, 0.0, -1.0]], dtype=torch.float64)


@pytest.fixture
def matrix() -> torch.Tensor:
    """Small SPD matrix matching `snapshots`' width."""
    return torch.eye(3, dtype=torch.float64) * 2.0


class TestWeightingNeedsMatrix:
    def test_raw_does_not_need_matrix(self) -> None:
        assert weighting_needs_matrix(RawWeightingConfig()) is False

    def test_power_norm_l2_does_not_need_matrix(self) -> None:
        assert weighting_needs_matrix(PowerNormWeightingConfig(metric="l2")) is False

    def test_power_norm_a_needs_matrix(self) -> None:
        assert weighting_needs_matrix(PowerNormWeightingConfig(metric="a")) is True

    def test_smoother_persistence_needs_matrix(self) -> None:
        assert weighting_needs_matrix(SmootherPersistenceWeightingConfig()) is True


class TestResolveRowScales:
    def test_raw_returns_none(self, snapshots: torch.Tensor) -> None:
        """`RawWeightingConfig` must resolve to `None` — POD's true no-op fit path."""
        assert resolve_row_scales(RawWeightingConfig(), snapshots, matrix=None) is None

    def test_power_norm_dispatches_to_torchalg_function(self, snapshots: torch.Tensor) -> None:
        cfg = PowerNormWeightingConfig(metric="l2", beta=0.5)
        scales = resolve_row_scales(cfg, snapshots, matrix=None)
        expected = power_norm_scales(snapshots, metric="l2", beta=0.5)
        torch.testing.assert_close(scales, expected)

    def test_power_norm_a_dispatches_with_matrix(
        self, snapshots: torch.Tensor, matrix: torch.Tensor
    ) -> None:
        cfg = PowerNormWeightingConfig(metric="a", beta=1.0)
        scales = resolve_row_scales(cfg, snapshots, matrix=matrix)
        expected = power_norm_scales(snapshots, matrix=matrix, metric="energy", beta=1.0)
        torch.testing.assert_close(scales, expected)

    def test_smoother_persistence_dispatches_to_torchalg_function(
        self, snapshots: torch.Tensor, matrix: torch.Tensor
    ) -> None:
        cfg = SmootherPersistenceWeightingConfig(omega=0.5, steps=3)
        scales = resolve_row_scales(cfg, snapshots, matrix=matrix)
        expected = smoother_persistence_scales(snapshots, matrix, omega=0.5, steps=3)
        torch.testing.assert_close(scales, expected)

    def test_power_norm_a_without_matrix_raises(self, snapshots: torch.Tensor) -> None:
        cfg = PowerNormWeightingConfig(metric="a", beta=1.0)
        with pytest.raises(ValueError, match="matrix"):
            resolve_row_scales(cfg, snapshots, matrix=None)

    def test_smoother_persistence_without_matrix_raises(self, snapshots: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="matrix"):
            resolve_row_scales(SmootherPersistenceWeightingConfig(), snapshots, matrix=None)
