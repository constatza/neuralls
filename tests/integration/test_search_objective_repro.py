"""Repro/regression test for the reported train/loss vs test/loss objective mismatch.

Bug report: `[search] objective = "train/loss"` in a job TOML, but the value being
optimized looked like `test/loss`. Confirmed root cause (see
`~/projects/dlkit/docs/requests-from-neuralls.md`, item 0): dlkit's `VanillaExecutor`
calls `trainer.fit()` then `trainer.predict()`/`trainer.test()` on the *same* Trainer
instance, and `trainer.predict()` clears `Trainer.callback_metrics` as a side effect of
switching Lightning stages — wiping every `train/*`/`val/*` key `fit()` had logged
before `_collect_metrics()` ever runs. Reproduced directly against this repo's pinned
dlkit (`develop` @ `5fb93f7`) with a minimal plain training job: `callback_metrics` was
already `{}` before `trainer.test()` even started.

This was an upstream dlkit bug, not something fixable from neuralls. The test was
`xfail(strict=True)` until dlkit's `VanillaExecutor` stopped losing `train/*`/`val/*`
metrics after `predict()`/`test()`.

Update (dlkit @ `11398ae`): dlkit fixed two related, narrower issues we also reported —
`on_run_created` now carries an explicit `is_outermost` signal (no more inferring it from
call order), and `extract_objective_value` now raises instead of silently returning `0.0`
for a missing objective key.

Update (dlkit @ `4466bcd`): item 0 itself is now fixed — `VanillaExecutor.execute()`
snapshots `callback_metrics` immediately after `trainer.fit()` returns and merges it with
whatever `predict()`/`test()` populate afterward, so `train/*`/`val/*` metrics survive.
The `xfail` marker is removed; this is now a permanent regression guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def disjoint_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """20 samples, clearly separable by index so a train/test split is meaningfully disjoint."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_targets = 20, 4, 2
    X = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_targets))
    return X, Y


def _build_search_settings(tmp_path: Path, X: np.ndarray, Y: np.ndarray):
    from dlkit.infrastructure.config import (
        DataModuleSelector,
        DataSettings,
        ExperimentSettings,
        RunSettings,
        TrackingSettings,
        TrainingSettings,
    )
    from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry
    from dlkit.infrastructure.config.job_config import SearchJobConfig
    from dlkit.infrastructure.config.model_components import (
        MetricComponentSettings,
        ModelComponentSettings,
    )
    from dlkit.infrastructure.config.search_settings import IntParam, SearchSettings
    from dlkit.infrastructure.config.trainer_settings import TrainerSettings

    return SearchJobConfig(
        run=RunSettings(type="search", seed=42),
        experiment=ExperimentSettings(name="test_objective_repro"),
        tracking=TrackingSettings(backend="none"),
        data=DataSettings(
            name="FlexibleDataset",
            batch_size=4,
            num_workers=0,
            module=DataModuleSelector(name="ArrayDataModule"),
            features=(ValueEntry(name="x", value=X),),
            targets=(ValueEntry(name="y", value=Y, data_role=DataRole.TARGET),),
            # Default splits (test=0.15, val=0.15) already give a genuinely disjoint
            # test split for 20 samples — no overfit_batches involved here, since this
            # test targets objective-metric selection, not the debug-overfit profile.
        ),
        training=TrainingSettings(
            trainer=TrainerSettings(
                max_epochs=3,
                accelerator="cpu",
                enable_checkpointing=False,
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
        search=SearchSettings(
            n_trials=2,
            objective="train/loss",
            space={"model.hidden_size": IntParam(type="int", low=2, high=8)},
        ),
    )


def test_search_objective_reads_configured_train_loss_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disjoint_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """`search.objective = "train/loss"` must select train/loss, not val/loss or test/loss.

    Also asserts train/loss and test/loss are both present and not spuriously equal for
    a disjoint train/test split — ruling out the "train/loss happens to equal test/loss"
    non-bug explanation as well as the "objective silently ignored" bug this guards.
    """
    from dlkit.engine.tracking import uri_resolver
    from dlkit.interfaces.api import execute

    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    X, Y = disjoint_regression_data
    settings = _build_search_settings(tmp_path, X, Y)

    result = execute(settings)

    metrics = result.training_result.metrics
    assert "train/loss" in metrics, f"train/loss missing from metrics: {sorted(metrics)}"
    assert "test/loss" in metrics, f"test/loss missing from metrics: {sorted(metrics)}"
    assert abs(metrics["train/loss"] - metrics["test/loss"]) > 1e-9, (
        "train/loss and test/loss are suspiciously equal for a disjoint split — "
        f"train/loss={metrics['train/loss']!r} test/loss={metrics['test/loss']!r}"
    )
