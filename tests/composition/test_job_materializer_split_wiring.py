"""materialize_training_job wires split pinning into its patch-stage chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry
from dlkit.infrastructure.config.data_settings import DataModuleSelector, DataSettings
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings

from neuralls.composition.assignments import job_materializer
from neuralls.composition.assignments.job_materializer import materialize_training_job
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.models.workspace import AssignmentWorkspace


def _settings(tmp_path: Path) -> TrainingJobConfig:
    rng = np.random.default_rng(0)
    x = rng.random((10, 2))
    y = rng.random((10, 1))
    return TrainingJobConfig(
        run=RunSettings(type="train", seed=42),
        data=DataSettings(
            name="FlexibleDataset",
            batch_size=4,
            num_workers=0,
            module=DataModuleSelector(name="ArrayDataModule"),
            features=(ValueEntry(name="x", value=x),),
            targets=(ValueEntry(name="y", value=y, data_role=DataRole.TARGET),),
        ),
        model=ModelComponentSettings(
            name="FFNN", module_path="dlkit.nn", hidden_size=4, num_layers=1
        ),
        training=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                default_root_dir=tmp_path,
            )
        ),
    )


def test_materialize_training_job_invokes_split_patch_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    original = job_materializer.patch_split_settings

    def _spy(settings: TrainingJobConfig, *, dataset_dir: Path) -> TrainingJobConfig:
        calls.append(dataset_dir)
        return original(settings, dataset_dir=dataset_dir)

    split_path = tmp_path / "data" / "split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("{}")
    monkeypatch.setattr(
        "neuralls.composition.assignments.runtime_split_patcher.resolve_or_create_pinned_split",
        lambda *_args, **_kwargs: split_path,
    )
    monkeypatch.setattr(job_materializer, "patch_split_settings", _spy)

    root_dir = tmp_path / "root"
    root_dir.mkdir(parents=True, exist_ok=True)
    workspace = AssignmentWorkspace(
        dataset_id="ds",
        run_id="run",
        root_dir=root_dir,
        data_dir=tmp_path / "data",
    )
    settings, returned_workspace = materialize_training_job(
        _settings(tmp_path),
        workspace=workspace,
        features=[],
        targets=[],
        contract=default_training_dataset_contract(),
        dataset_format="hdf5",
    )

    assert calls == [workspace.data_dir]
    assert returned_workspace is workspace
    assert settings.data is not None
    assert str(settings.data.splits.filepath) == str(split_path)
