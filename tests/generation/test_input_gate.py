"""Comparison refuses datasets that generation has not produced and pinned under today's config."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neuralls.composition.comparison import config_assembler, input_gate
from neuralls.composition.comparison.input_gate import require_current_inputs
from neuralls.composition.generation.processing import process_config
from neuralls.composition.generation.split_pinning import pin_dataset_split
from neuralls.composition.identity.generation import generation_identity
from neuralls.platform.config.models.data_models import DataConfigFile, SplitConfig, StrategyConfig
from neuralls.shared.constants import REGENERATE_DATASETS_COMMAND, SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale
from tests.dataset_support import strip_provenance

type ConfigFactory = Callable[..., DataConfigFile]


@pytest.fixture
def served_config(monkeypatch: pytest.MonkeyPatch) -> list[DataConfigFile]:
    """The gate reads whichever config this list holds, standing in for the registry."""
    served: list[DataConfigFile] = []
    for module in (input_gate, config_assembler):
        monkeypatch.setattr(module, "resolve_dataset_config_path", lambda *_args: Path("ds.toml"))
        monkeypatch.setattr(module, "load_data_config", lambda *_args: served[0])
    return served


def _gate(config: DataConfigFile) -> None:
    require_current_inputs(MagicMock(), Path("."), MagicMock(), [config.id])


def _generate_and_pin(config: DataConfigFile) -> Path:
    matrix = np.loadtxt(config.source.matrix_path)
    dataset_dir = process_config(config, matrix, identity=generation_identity(config))
    pin_dataset_split(dataset_dir, config.split)
    return dataset_dir


def test_a_generated_and_pinned_dataset_passes(
    data_config: DataConfigFile, served_config: list[DataConfigFile]
) -> None:
    served_config.append(data_config)
    _generate_and_pin(data_config)

    _gate(data_config)


def test_an_ungenerated_dataset_asks_for_generation(
    data_config: DataConfigFile, served_config: list[DataConfigFile]
) -> None:
    served_config.append(data_config)

    with pytest.raises(StageInputsStale):
        _gate(data_config)


def test_a_dataset_whose_sources_changed_is_stale(
    data_config: DataConfigFile, served_config: list[DataConfigFile], spd_matrix_file: Path
) -> None:
    served_config.append(data_config)
    _generate_and_pin(data_config)

    np.savetxt(spd_matrix_file, np.eye(4) * 7.0)

    with pytest.raises(StageInputsStale):
        _gate(data_config)


def test_a_missing_split_asks_for_generation(
    data_config: DataConfigFile, served_config: list[DataConfigFile]
) -> None:
    served_config.append(data_config)
    dataset_dir = _generate_and_pin(data_config)

    (dataset_dir / SPLIT_FILENAME).unlink()

    with pytest.raises(StageInputsStale):
        _gate(data_config)


def test_a_split_config_change_is_stale_without_regenerating_the_dataset(
    data_config: DataConfigFile, served_config: list[DataConfigFile]
) -> None:
    ten_rows = data_config.model_copy(
        update={
            "generation": data_config.generation.model_copy(
                update={"strategy": [StrategyConfig(name="neutral_ones", samples=10)]}
            )
        }
    )
    _generate_and_pin(ten_rows)
    changed = ten_rows.model_copy(update={"split": SplitConfig(test_ratio=0.5, val_ratio=0.0)})
    served_config.append(changed)

    # The dataset itself is still current: [split] is not part of generation identity.
    assert generation_identity(changed).key == generation_identity(ten_rows).key
    with pytest.raises(StageInputsStale):
        _gate(changed)


def test_a_dataset_generated_before_provenance_existed_is_stale_and_names_the_command(
    data_config: DataConfigFile, served_config: list[DataConfigFile]
) -> None:
    served_config.append(data_config)
    dataset_dir = _generate_and_pin(data_config)
    strip_provenance(dataset_dir)

    # Identity is unchanged, so generation alone would skip it: only --force regenerates.
    with pytest.raises(StageInputsStale, match="no scientific provenance") as raised:
        _gate(data_config)
    assert str(raised.value) == (
        f"Dataset 'ds' at '{dataset_dir}' has no scientific provenance (it was generated before "
        f"provenance was recorded): run `{REGENERATE_DATASETS_COMMAND}` to rewrite it."
    )
    assert REGENERATE_DATASETS_COMMAND == "neuralls generate --force <case-config.toml>"
