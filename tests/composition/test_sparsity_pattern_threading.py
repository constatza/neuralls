"""``[output].sparsity_pattern`` reaches ``write_csr_streamed`` through the config entry point.

The spy records the ``sparsity_pattern`` argument and forwards the call unchanged, so the
dataset is still written. Matrix sources and configs come from fixtures in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation import dataset_builder
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.types import DatasetFormat, SparsityPattern

_DATASET_ID = "sparsity-threading"
_MakeConfig = Callable[[str | None], Path]


@pytest.fixture
def make_csr_config(tmp_path: Path) -> _MakeConfig:
    """Build a single-matrix CSR config; ``pattern_line`` is the ``[output]`` extra line or None."""
    matrices_dir = tmp_path / "matrices"
    matrices_dir.mkdir()
    np.savetxt(matrices_dir / "matrix_only.txt", np.eye(2) * 2.0)

    def _make(pattern_line: str | None) -> Path:
        extra = "" if pattern_line is None else f'sparsity_pattern = "{pattern_line}"\n'
        config_path = tmp_path / f"{_DATASET_ID}.toml"
        config_path.write_text(
            f"""
id = "{_DATASET_ID}"

[source]
matrix_path = "{(matrices_dir / "matrix_*.txt").as_posix()}"
enumerate_by = "name"

[generation]
normalize = "none"

[[generation.strategy]]
name = "neutral_ones"
samples = 2

[output]
matrix_format = "csr"
{extra}"""
        )
        return config_path

    return _make


@pytest.fixture
def recorded_patterns(monkeypatch: pytest.MonkeyPatch) -> list[SparsityPattern]:
    """Record the ``sparsity_pattern`` every streamed CSR build receives."""
    patterns: list[SparsityPattern] = []
    original = dataset_builder.write_csr_streamed

    def _recording(
        source: SourceSpec,
        spec: DatasetSpec,
        dataset_dir: Path,
        dataset_format: DatasetFormat,
        *,
        sparsity_pattern: SparsityPattern,
    ) -> None:
        patterns.append(sparsity_pattern)
        return original(
            source, spec, dataset_dir, dataset_format, sparsity_pattern=sparsity_pattern
        )

    monkeypatch.setattr(dataset_builder, "write_csr_streamed", _recording)
    return patterns


def test_shared_pattern_setting_reaches_streamed_csr_writer(
    make_csr_config: _MakeConfig,
    recorded_patterns: list[SparsityPattern],
    neuralls_settings: NeurallsSettings,
) -> None:
    process_data_from_config(make_csr_config("shared"), neuralls_settings)
    assert recorded_patterns == [SparsityPattern.SHARED]


def test_default_sparsity_pattern_reaches_streamed_csr_writer_as_ragged(
    make_csr_config: _MakeConfig,
    recorded_patterns: list[SparsityPattern],
    neuralls_settings: NeurallsSettings,
) -> None:
    process_data_from_config(make_csr_config(None), neuralls_settings)
    assert recorded_patterns == [SparsityPattern.RAGGED]
