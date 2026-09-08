"""Tests for case-config-driven MLflow injection."""

from __future__ import annotations

from pathlib import Path

import tomli_w

from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment


def test_load_experiment_injects_mlflow_from_case_config(
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """Job configs without tracking config get runtime-injected topology."""
    job_config_path = tmp_path / "job.toml"
    model_profile_path = tmp_path / "model.toml"
    data_path = tmp_path / "data.toml"
    experiments_path = tmp_path / "experiments.toml"

    mlflow_db = tmp_path / "mlruns" / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    model_profile = {
        "model": {"name": "LinearNetwork", "module_path": "dlkit.nn"},
        "data": {
            "name": "FlexibleDataset",
            "module": {"name": "ArrayDataModule"},
        },
    }
    with open(model_profile_path, "wb") as fh:
        tomli_w.dump(model_profile, fh)

    job_config = {
        "run": {
            "type": "train",
            "seed": 42,
            "model": model_profile_path.name,
            "data": model_profile_path.name,
        },
        "experiment": {"name": "SystemInjected"},
        "training": {
            "trainer": {"max_epochs": 1, "accelerator": "cpu"},
            "optimizer": {
                "default_optimizer": {"name": "AdamW", "lr": 1e-3},
            },
        },
    }
    with open(job_config_path, "wb") as fh:
        tomli_w.dump(job_config, fh)

    data_config = {
        "id": "system-injected-data",
        "source": {},
        "generation": {},
        "output": {"data_dir": str(tmp_path / "processed/system-injected-data")},
    }
    with open(data_path, "wb") as fh:
        tomli_w.dump(data_config, fh)

    experiments_config = {
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
        "names": {
            "training": "Train",
            "comparison": "Comparisons",
        },
    }
    with open(experiments_path, "wb") as fh:
        tomli_w.dump(experiments_config, fh)

    experiment = load_assignment(
        job_config_path=job_config_path,
        data_config_path=data_path,
        neuralls_settings=neuralls_settings,
        case_config_path=experiments_path,
        identity=AssignmentIdentity(dataset_registry_id=data_path.stem),
    )

    assert experiment.settings.tracking is not None
    assert experiment.settings.tracking.backend == "mlflow"
