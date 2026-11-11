"""A dataset config names its source problem, as a plain label, before generation starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation._context_builder import _build_context
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.shared.scientific_description.models import ScientificDescription

DATA_DIR = Path("datasets")


def test_source_problem_name_must_be_a_plain_label() -> None:
    from pydantic import ValidationError

    from neuralls.platform.config.models.data_models import SourceConfig

    assert SourceConfig(matrix_path="m.mtx", problem="Spectral 3D field").problem
    with pytest.raises(ValidationError):
        SourceConfig(matrix_path="m.mtx", problem="spectral_3d_field")


def _config(problem: str | None) -> DataConfigFile:
    """A resolved config; the matrix path and output directory are only stored, never opened."""
    return DataConfigFile(
        id="ds",
        source=SourceConfig(matrix_path="unopened/*.mtx", problem=problem),
        generation=GenerationConfig(
            normalize="none", strategy=[StrategyConfig(name="neutral_ones", samples=1)]
        ),
        output=OutputConfig(data_dir=DATA_DIR),
    )


def test_context_carries_the_configured_problem_as_provenance() -> None:
    named, _ = _build_context(config=_config("Spectral 3D field"))

    assert named.source_provenance == DatasetSourceProvenance(
        ScientificDescription("Spectral 3D field")
    )


def test_a_config_without_a_problem_cannot_start_generation() -> None:
    """Kills: warning and writing a dataset that can never be described."""
    with pytest.raises(ValueError, match=r"Dataset config 'ds' has no '\[source\] problem'"):
        _build_context(config=_config(None))
