"""The split patch stage points jobs at the pinned split and never creates one."""

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

from neuralls.composition.assignments._job_types import TrainableJobConfig
from neuralls.composition.assignments.runtime_split_patcher import patch_split_settings
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale


def _data_settings() -> DataSettings:
    rng = np.random.default_rng(0)
    return DataSettings(
        name="FlexibleDataset",
        batch_size=4,
        num_workers=0,
        module=DataModuleSelector(name="ArrayDataModule"),
        features=(ValueEntry(name="x", value=rng.random((10, 2))),),
        targets=(ValueEntry(name="y", value=rng.random((10, 1)), data_role=DataRole.TARGET),),
        splits=IndexSplitSettings(test=0.3, val=0.25),
    )


def _model_settings() -> ModelComponentSettings:
    return ModelComponentSettings(name="FFNN", module_path="dlkit.nn", hidden_size=4, num_layers=1)


@pytest.fixture
def training_job(tmp_path: Path) -> TrainingJobConfig:
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


@pytest.fixture
def fit_job() -> FitJobConfig:
    return FitJobConfig(
        run=RunSettings(type="fit", seed=42), data=_data_settings(), model=_model_settings()
    )


@pytest.fixture
def pinned_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / SPLIT_FILENAME).write_text("{}", encoding="utf-8")
    return dataset_dir


@pytest.mark.parametrize("job_fixture", ["training_job", "fit_job"])
def test_every_job_kind_reads_the_pinned_split(
    job_fixture: str, pinned_dataset_dir: Path, request: pytest.FixtureRequest
) -> None:
    job: TrainableJobConfig = request.getfixturevalue(job_fixture)

    patched = patch_split_settings(job, dataset_dir=pinned_dataset_dir)

    assert patched.data is not None
    assert str(patched.data.splits.filepath) == str(pinned_dataset_dir / SPLIT_FILENAME)


def test_missing_split_asks_for_generation_instead_of_creating_one(
    training_job: TrainingJobConfig, tmp_path: Path
) -> None:
    with pytest.raises(StageInputsStale):
        patch_split_settings(training_job, dataset_dir=tmp_path)

    assert not (tmp_path / SPLIT_FILENAME).exists()
