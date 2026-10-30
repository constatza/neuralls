"""Tests for the config-to-torchalg weighting bridge (`composition/preconditioners/_weighting.py`).

The actual weighting math lives in `torchalg.preconditioners.implementations.pod.weighting`
(tested there); this module's only job is dispatching a `SnapshotWeightingConfig`
to the right torchalg function, so these tests only need to confirm that
dispatch — not re-verify the math.
"""

from __future__ import annotations

import pytest
import torch

from neuralls.composition.preconditioners import _weighting as pod_module
from neuralls.composition.preconditioners._weighting import (
    resolve_row_scales,
    weighting_needs_matrix,
)
from neuralls.platform.config.models.weighting import (
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

    def test_power_norm_dispatches_to_torchalg_function(
        self, monkeypatch: pytest.MonkeyPatch, snapshots: torch.Tensor
    ) -> None:
        cfg = PowerNormWeightingConfig(metric="l2", beta=0.5)
        expected = torch.tensor([3.0], dtype=torch.float64)
        calls: list[tuple[torch.Tensor, torch.Tensor | None, str, float]] = []

        def fake_power_norm_scales(
            values: torch.Tensor,
            matrix: torch.Tensor | None = None,
            *,
            metric: str,
            beta: float,
        ) -> torch.Tensor:
            calls.append((values, matrix, metric, beta))
            return expected

        monkeypatch.setattr(pod_module, "power_norm_scales", fake_power_norm_scales)

        scales = resolve_row_scales(cfg, snapshots, matrix=None)

        assert scales is expected
        assert calls == [(snapshots, None, "l2", 0.5)]

    def test_power_norm_a_dispatches_with_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        snapshots: torch.Tensor,
        matrix: torch.Tensor,
    ) -> None:
        cfg = PowerNormWeightingConfig(metric="a", beta=1.0)
        expected = torch.tensor([4.0], dtype=torch.float64)
        calls: list[tuple[torch.Tensor, torch.Tensor | None, str, float]] = []

        def fake_power_norm_scales(
            values: torch.Tensor,
            matrix: torch.Tensor | None = None,
            *,
            metric: str,
            beta: float,
        ) -> torch.Tensor:
            calls.append((values, matrix, metric, beta))
            return expected

        monkeypatch.setattr(pod_module, "power_norm_scales", fake_power_norm_scales)

        scales = resolve_row_scales(cfg, snapshots, matrix=matrix)

        assert scales is expected
        assert calls == [(snapshots, matrix, "energy", 1.0)]

    def test_smoother_persistence_dispatches_to_torchalg_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
        snapshots: torch.Tensor,
        matrix: torch.Tensor,
    ) -> None:
        cfg = SmootherPersistenceWeightingConfig(omega=0.5, steps=3)
        expected = torch.tensor([5.0], dtype=torch.float64)
        calls: list[tuple[torch.Tensor, torch.Tensor, float, int]] = []

        def fake_smoother_persistence_scales(
            values: torch.Tensor,
            candidate_matrix: torch.Tensor,
            *,
            omega: float,
            steps: int,
        ) -> torch.Tensor:
            calls.append((values, candidate_matrix, omega, steps))
            return expected

        monkeypatch.setattr(
            pod_module, "smoother_persistence_scales", fake_smoother_persistence_scales
        )

        scales = resolve_row_scales(cfg, snapshots, matrix=matrix)

        assert scales is expected
        assert calls == [(snapshots, matrix, 0.5, 3)]

    def test_power_norm_a_without_matrix_raises(self, snapshots: torch.Tensor) -> None:
        cfg = PowerNormWeightingConfig(metric="a", beta=1.0)
        with pytest.raises(ValueError):
            resolve_row_scales(cfg, snapshots, matrix=None)

    def test_smoother_persistence_without_matrix_raises(self, snapshots: torch.Tensor) -> None:
        with pytest.raises(ValueError):
            resolve_row_scales(SmootherPersistenceWeightingConfig(), snapshots, matrix=None)
