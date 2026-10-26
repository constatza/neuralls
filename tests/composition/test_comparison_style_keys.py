"""Tests for `_pod2g_style_keys`, the plot-style-key extractor for POD-2G comparisons.

A dense POD-2G sweep (many fit-dataset x weighting-scheme combinations in one
comparison plot) needs a color/marker key per entry beyond just its family —
see `neuralls.platform.reporting.plots._resolve_styles`. These pin what
`_pod2g_style_keys` extracts from a raw preconditioner config, and that it
stays `(None, None)` for anything that isn't POD-2G.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.comparison.comparison_run import _pod2g_style_keys
from neuralls.composition.preconditioners.labels import build_configured_preconditioner_label
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PowerNormWeightingConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.scientific_description import NotationPolicy, RenderTarget


@pytest.fixture
def pod2g_dataset_dir(tmp_path: Path) -> Path:
    """A stand-in POD-2G fit-dataset directory (no validator requires it exist yet)."""
    return tmp_path / "gaussian-cg0-rectangular-high-condition"


@pytest.fixture
def pod2g_config_with_power_norm_weighting(pod2g_dataset_dir: Path) -> AMGPreconditionerConfig:
    """An AMG config with POD-2G coarsening and a non-default weighting scheme."""
    coarsening = PODCoarseningConfig(
        dataset_dir=pod2g_dataset_dir,
        rank=8,
        weighting=PowerNormWeightingConfig(metric="a", beta=1.0),
    )
    return AMGPreconditionerConfig(coarsening=coarsening)


@pytest.fixture
def neural_pod2g_config(pod2g_dataset_dir: Path) -> AMGPreconditionerConfig:
    """An AMG config with checkpoint-predicted (neural) POD-2G coarsening."""
    coarsening = NeuralPODCoarseningConfig(dataset_dir=pod2g_dataset_dir, input_names=("params",))
    return AMGPreconditionerConfig(coarsening=coarsening)


@pytest.fixture
def jacobi_config() -> StandardPreconditionerConfig:
    """A non-AMG preconditioner config — no coarsening/dataset concept at all."""
    return StandardPreconditionerConfig(type=PreconditionerType.JACOBI)


def test_pod2g_style_keys_extracts_dataset_and_weighting(
    pod2g_config_with_power_norm_weighting: AMGPreconditionerConfig,
    pod2g_dataset_dir: Path,
) -> None:
    """A PODCoarseningConfig yields its fit dataset as color_key, weighting method as marker_key."""
    color_key, marker_key = _pod2g_style_keys(pod2g_config_with_power_norm_weighting)
    assert color_key == str(pod2g_dataset_dir)
    assert marker_key == "power_norm"


def test_pod2g_style_keys_neural_pod_has_color_but_no_marker_key(
    neural_pod2g_config: AMGPreconditionerConfig,
    pod2g_dataset_dir: Path,
) -> None:
    """NeuralPODCoarseningConfig has a fit dataset but no weighting scheme to key on."""
    color_key, marker_key = _pod2g_style_keys(neural_pod2g_config)
    assert color_key == str(pod2g_dataset_dir)
    assert marker_key is None


def test_pod2g_style_keys_non_pod_config_falls_back_to_none(
    jacobi_config: StandardPreconditionerConfig,
) -> None:
    """A non-POD-2G config yields (None, None), falling back to family-based styling."""
    assert _pod2g_style_keys(jacobi_config) == (None, None)


def test_configured_pod_label_honours_notation_without_building(
    pod2g_config_with_power_norm_weighting: AMGPreconditionerConfig,
) -> None:
    """Compatibility labels retain POD family and notation from config-only facts."""
    config = pod2g_config_with_power_norm_weighting.model_copy(update={"pre_smoothing_steps": 3})

    label = build_configured_preconditioner_label(
        config,
        policy=NotationPolicy.default(RenderTarget.MATHTEXT),
    )

    assert label.startswith("POD-2G")
    assert r"\nu_1=3" in label


def test_configured_pod_labels_expose_rank_without_config_slug(pod2g_dataset_dir: Path) -> None:
    """POD variants remain concise and distinguish their configured ranks."""
    rank_three = build_configured_preconditioner_label(
        AMGPreconditionerConfig(
            coarsening=PODCoarseningConfig(dataset_dir=pod2g_dataset_dir, rank=3)
        )
    )
    rank_five = build_configured_preconditioner_label(
        AMGPreconditionerConfig(
            coarsening=PODCoarseningConfig(dataset_dir=pod2g_dataset_dir, rank=5)
        )
    )

    assert rank_three == "POD-2G (r=3, ν₁=2, ν₂=2, ωₛ=auto)"
    assert rank_five == "POD-2G (r=5, ν₁=2, ν₂=2, ωₛ=auto)"
    assert "config=" not in rank_three
