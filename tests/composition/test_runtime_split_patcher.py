"""Split-pinning runtime patch stage dispatches ratios correctly by job kind."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry
from dlkit.infrastructure.config.data_settings import DataModuleSelector, DataSettings
from dlkit.infrastructure.config.job_config import FitJobConfig, TrainingJobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.split_settings import IndexSplitSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings

from neuralls.composition.assignments import runtime_split_patcher
from neuralls.composition.assignments.runtime_split_patcher import patch_split_settings
from neuralls.shared.constants import DEFAULT_SPLIT_SEED

_TEST_RATIO = 0.3
_VAL_RATIO = 0.25


def _data_settings() -> DataSettings:
    rng = np.random.default_rng(0)
    x = rng.random((10, 2))
    y = rng.random((10, 1))
    return DataSettings(
        name="FlexibleDataset",
        batch_size=4,
        num_workers=0,
        module=DataModuleSelector(name="ArrayDataModule"),
        features=(ValueEntry(name="x", value=x),),
        targets=(ValueEntry(name="y", value=y, data_role=DataRole.TARGET),),
        splits=IndexSplitSettings(test=_TEST_RATIO, val=_VAL_RATIO),
    )


def _model_settings() -> ModelComponentSettings:
    return ModelComponentSettings(name="FFNN", module_path="dlkit.nn", hidden_size=4, num_layers=1)


def _training_job_config(tmp_path: Path) -> TrainingJobConfig:
    return TrainingJobConfig(
        run=RunSettings(type="train", seed=42),
        data=_data_settings(),
        model=_model_settings(),
        training=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                default_root_dir=tmp_path,
            )
        ),
    )


def _fit_job_config() -> FitJobConfig:
    return FitJobConfig(
        run=RunSettings(type="fit", seed=42),
        data=_data_settings(),
        model=_model_settings(),
    )


@pytest.fixture
def pinned_split_spy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")

    def _fake_resolve(dataset_dir: Path, *, test_ratio: float, val_ratio: float, seed: int) -> Path:
        calls.append(
            {
                "dataset_dir": dataset_dir,
                "test_ratio": test_ratio,
                "val_ratio": val_ratio,
                "seed": seed,
            }
        )
        return split_path

    monkeypatch.setattr(runtime_split_patcher, "resolve_or_create_pinned_split", _fake_resolve)
    return calls


def test_training_job_uses_configured_val_ratio(
    tmp_path: Path, pinned_split_spy: list[dict[str, object]]
) -> None:
    settings = _training_job_config(tmp_path)

    patched = patch_split_settings(settings, dataset_dir=tmp_path)

    assert pinned_split_spy == [
        {
            "dataset_dir": tmp_path,
            "test_ratio": _TEST_RATIO,
            "val_ratio": _VAL_RATIO,
            "seed": DEFAULT_SPLIT_SEED,
        }
    ]
    assert patched.data is not None
    assert str(patched.data.splits.filepath) == str(tmp_path / "split.json")


def test_fit_job_forces_val_ratio_to_zero(
    tmp_path: Path, pinned_split_spy: list[dict[str, object]]
) -> None:
    settings = _fit_job_config()

    patch_split_settings(settings, dataset_dir=tmp_path)

    assert pinned_split_spy == [
        {
            "dataset_dir": tmp_path,
            "test_ratio": _TEST_RATIO,
            "val_ratio": 0.0,
            "seed": DEFAULT_SPLIT_SEED,
        }
    ]
