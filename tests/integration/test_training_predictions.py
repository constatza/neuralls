"""A fast-dev-run DLKit training run captures stacked predictions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dlkit.common.results import TrainingResult
from dlkit.config import TrainingJobConfig
from dlkit.engine.tracking import uri_resolver
from dlkit.infrastructure.config import (
    DataModuleSelector,
    DataSettings,
    ExperimentSettings,
    RunSettings,
    TrackingSettings,
    TrainingSettings,
)
from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry
from dlkit.infrastructure.config.model_components import (
    MetricComponentSettings,
    ModelComponentSettings,
)
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.interfaces.api import execute


def test_fast_dev_run_captures_stacked_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainer.predict() produces stacked TensorDict and to_numpy() output.

    This test documents the expected API so _log_training_evaluation() can rely on it.
    Runs a single fast_dev_run training step using DLKit programmatic settings.
    """
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    n_samples, n_features, n_targets = 20, 4, 2
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_targets))

    settings = TrainingJobConfig(
        run=RunSettings(type="train", seed=42),
        experiment=ExperimentSettings(name="test_predict_structure"),
        tracking=TrackingSettings(backend="none"),
        data=DataSettings(
            name="FlexibleDataset",
            batch_size=4,
            num_workers=0,
            module=DataModuleSelector(name="ArrayDataModule"),
            features=(ValueEntry(name="x", value=X),),
            targets=(ValueEntry(name="y", value=Y, data_role=DataRole.TARGET),),
        ),
        training=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                default_root_dir=tmp_path,
            ),
            metrics=(
                MetricComponentSettings(
                    name="MeanSquaredError",
                    module_path="dlkit.domain.metrics",
                ),
            ),
        ),
        model=ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.nn",
            hidden_size=4,
            num_layers=1,
        ),
    )

    result = cast(TrainingResult, execute(settings))

    assert result.predictions is not None, "trainer.predict() should have captured predictions"
    assert isinstance(result.predictions, list)
    assert len(result.predictions) > 0

    stacked = result.stacked
    assert stacked is not None, "result.stacked should be a TensorDict after prediction"

    all_numpy = result.to_numpy()
    assert all_numpy is not None, "to_numpy() should return a dict after prediction"
    assert "predictions" in all_numpy, f"Expected 'predictions' key, got {list(all_numpy.keys())}"
    preds = all_numpy["predictions"]
    if isinstance(preds, dict):
        output = next(iter(preds.values()))
    else:
        output = preds
    assert isinstance(output, np.ndarray)
    assert output.ndim >= 1
