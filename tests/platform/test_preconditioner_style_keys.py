"""Tests for `pod_color_key`, the plot colour key of POD comparisons.

A POD sweep over many fit datasets in one comparison plot needs a colour key per
entry beyond just its family — see
`neuralls.platform.reporting.series_style.resolve_styles`. These pin what
`pod_color_key` extracts from a raw preconditioner config, and that it stays
`None` for anything that isn't POD.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.comparison.models import canonical_method_config
from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
    parse_preconditioner_config,
)
from neuralls.platform.config.models.preconditioner_family import (
    pod_color_key,
    preconditioner_family,
)


@pytest.fixture
def pod_dataset_dir() -> Path:
    """A stand-in POD fit-dataset directory (no validator requires it exist yet)."""
    return Path("workspace") / "gaussian-cg0-rectangular-high-condition"


@pytest.fixture
def pod_config(pod_dataset_dir: Path) -> AMGPreconditionerConfig:
    """An AMG config with POD coarsening fit inline from a dataset."""
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=pod_dataset_dir, rank=8)
    )


@pytest.fixture
def neural_pod_config(pod_dataset_dir: Path) -> AMGPreconditionerConfig:
    """An AMG config with checkpoint-predicted (neural) POD coarsening."""
    coarsening = NeuralPODCoarseningConfig(dataset_dir=pod_dataset_dir, input_names=("params",))
    return AMGPreconditionerConfig(coarsening=coarsening)


@pytest.fixture
def jacobi_config() -> StandardPreconditionerConfig:
    """A non-AMG preconditioner config — no coarsening/dataset concept at all."""
    return StandardPreconditionerConfig(type=PreconditionerType.JACOBI)


def test_pod_color_key_is_the_fit_dataset(
    pod_config: AMGPreconditionerConfig, pod_dataset_dir: Path
) -> None:
    """A PODCoarseningConfig yields its fit dataset as colour key."""
    assert pod_color_key(pod_config) == str(pod_dataset_dir)


def test_neural_pod_color_key_is_the_fit_dataset(
    neural_pod_config: AMGPreconditionerConfig, pod_dataset_dir: Path
) -> None:
    """NeuralPODCoarseningConfig is keyed by its parameter dataset too."""
    assert pod_color_key(neural_pod_config) == str(pod_dataset_dir)


def test_pod_color_key_non_pod_config_falls_back_to_none(
    jacobi_config: StandardPreconditionerConfig,
) -> None:
    """A non-POD config yields None, falling back to family-based styling."""
    assert pod_color_key(jacobi_config) is None


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


def test_stored_config_keeps_family_and_missing_keys(
    fit_dataset_dirs: tuple[Path, Path, Path],
    jacobi_config: StandardPreconditionerConfig,
) -> None:
    """Family survives the persisted form; non-POD configs stay unkeyed."""
    pod = _pod(fit_dataset_dirs[0])
    neural_pod = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=fit_dataset_dirs[0], input_names=("params",)
        )
    )
    for config in (pod, neural_pod, jacobi_config):
        assert preconditioner_family(_stored(config)) == preconditioner_family(config)
    assert pod_color_key(_stored(jacobi_config)) is None


def test_stored_config_color_key_groups_by_fit_dataset_content(
    fit_dataset_dirs: tuple[Path, Path, Path],
) -> None:
    """The persisted config holds the fit dataset's content digest, so the colour key groups by content."""
    first, same_content, other = fit_dataset_dirs
    key = {path: pod_color_key(_stored(_pod(path))) for path in fit_dataset_dirs}
    assert key[first] == key[same_content]
    assert key[first] != key[other]
