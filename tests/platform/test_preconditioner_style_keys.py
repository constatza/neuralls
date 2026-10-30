"""Tests for `pod2g_style_keys`, the plot-style-key extractor for POD-2G comparisons.

A dense POD-2G sweep (many fit-dataset x weighting-scheme combinations in one
comparison plot) needs a color/marker key per entry beyond just its family —
see `neuralls.platform.reporting.plots._resolve_styles`. These pin what
`pod2g_style_keys` extracts from a raw preconditioner config, and that it
stays `(None, None)` for anything that isn't POD-2G.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.comparison.models import canonical_method_config
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PowerNormWeightingConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
    parse_preconditioner_config,
)
from neuralls.platform.config.models.preconditioner_family import (
    pod2g_style_keys,
    preconditioner_family,
)


@pytest.fixture
def pod2g_dataset_dir() -> Path:
    """A stand-in POD-2G fit-dataset directory (no validator requires it exist yet)."""
    return Path("/workspace") / "gaussian-cg0-rectangular-high-condition"


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
    color_key, marker_key = pod2g_style_keys(pod2g_config_with_power_norm_weighting)
    assert color_key == str(pod2g_dataset_dir)
    assert marker_key == "power_norm"


def test_pod2g_style_keys_neural_pod_has_color_but_no_marker_key(
    neural_pod2g_config: AMGPreconditionerConfig,
    pod2g_dataset_dir: Path,
) -> None:
    """NeuralPODCoarseningConfig has a fit dataset but no weighting scheme to key on."""
    color_key, marker_key = pod2g_style_keys(neural_pod2g_config)
    assert color_key == str(pod2g_dataset_dir)
    assert marker_key is None


def test_pod2g_style_keys_non_pod_config_falls_back_to_none(
    jacobi_config: StandardPreconditionerConfig,
) -> None:
    """A non-POD-2G config yields (None, None), falling back to family-based styling."""
    assert pod2g_style_keys(jacobi_config) == (None, None)


@pytest.fixture
def fit_dataset_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two fit-dataset directories with identical content and one with different content."""
    first, same_content, other = (tmp_path / name for name in ("first", "same", "other"))
    for directory, content in ((first, "a"), (same_content, "a"), (other, "b")):
        directory.mkdir()
        (directory / "snapshots.txt").write_text(content, encoding="utf-8")
    return first, same_content, other


def _stored(config: PreconditionerConfig) -> PreconditionerConfig:
    """The config as a report sees it: parsed back from the persisted canonical form."""
    return parse_preconditioner_config(canonical_method_config(config))


def _pod(directory: Path) -> AMGPreconditionerConfig:
    return AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=directory, rank=8))


def test_stored_config_keeps_family_marker_and_missing_keys(
    fit_dataset_dirs: tuple[Path, Path, Path],
    jacobi_config: StandardPreconditionerConfig,
) -> None:
    """Family and marker survive the persisted form; non-POD configs stay unkeyed."""
    pod = _pod(fit_dataset_dirs[0])
    neural_pod = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=fit_dataset_dirs[0], input_names=("params",)
        )
    )
    for config in (pod, neural_pod, jacobi_config):
        assert preconditioner_family(_stored(config)) == preconditioner_family(config)
        assert pod2g_style_keys(_stored(config))[1] == pod2g_style_keys(config)[1]
    assert pod2g_style_keys(_stored(jacobi_config)) == (None, None)


def test_stored_config_color_key_groups_by_fit_dataset_content(
    fit_dataset_dirs: tuple[Path, Path, Path],
) -> None:
    """The persisted config holds the fit dataset's content digest, so the colour key groups by content."""
    first, same_content, other = fit_dataset_dirs
    key = {path: pod2g_style_keys(_stored(_pod(path)))[0] for path in fit_dataset_dirs}
    assert key[first] == key[same_content]
    assert key[first] != key[other]
