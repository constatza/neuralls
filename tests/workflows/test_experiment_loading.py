"""Tests for lower-case DLKit experiment configuration loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tomli_w


@pytest.fixture
def training_setup(tmp_path: Path) -> dict:
    """Create minimal training setup with data and configs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    jobs_dir = configs_dir / "jobs"
    jobs_dir.mkdir()

    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"

    n = 10
    A = np.random.rand(n, n)
    A = A.T @ A + np.eye(n)
    b = np.random.rand(n)

    np.savetxt(matrix_path, A)
    np.savetxt(rhs_path, b)

    return {
        "tmp_path": tmp_path,
        "data_dir": data_dir,
        "datasets_dir": datasets_dir,
        "jobs_dir": jobs_dir,
        "matrix_path": matrix_path,
        "rhs_path": rhs_path,
    }


def _write_training_job_config(
    path: Path,
    *,
    experiment_name: str,
    model_class: str,
    enable_tracking: bool = False,
) -> None:
    """Write a minimal lower-case training job plus its model profile."""
    model_profile_path = path.with_name(f"{path.stem}-profile.toml")
    model_profile = {
        "model": {
            "name": model_class,
            "module_path": "dlkit.nn",
            "hidden_size": 2,
            "num_layers": 1,
        },
        "data": {
            "name": "FlexibleDataset",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "module": {"name": "ArrayDataModule"},
        },
    }
    with open(model_profile_path, "wb") as f:
        tomli_w.dump(model_profile, f)

    job_config = {
        "run": {
            "type": "train",
            "seed": 42,
            "precision": "64",
            "model": model_profile_path.name,
            "data": model_profile_path.name,
        },
        "experiment": {
            "name": experiment_name,
        },
        "training": {
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": True,
                "num_sanity_val_steps": 0,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
            },
            "optimizer": {
                "default_optimizer": {"lr": 1e-3, "name": "AdamW"},
            },
            "metrics": [
                {
                    "name": "RelativeVectorNormError",
                    "module_path": "dlkit.domain.metrics",
                    "norm_ord": 2,
                    "vector_dim": -1,
                }
            ],
        },
    }
    if enable_tracking:
        job_config["tracking"] = {"backend": "mlflow"}
    with open(path, "wb") as f:
        tomli_w.dump(job_config, f)


class TestTrainingPipelineWithMLflow:
    """Workflow-level configuration loading checks."""

    def test_full_training_pipeline_with_mlflow(
        self,
        training_setup: dict,
        neuralls_settings,
    ) -> None:
        """MLflow-enabled jobs load with runtime tracking and workspace roots."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        jobs_dir = training_setup["jobs_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        data_config_path = datasets_dir / "mlflow_test_data.toml"
        with open(data_config_path, "wb") as f:
            tomli_w.dump(
                {
                    "id": "mlflow_test_data",
                    "source": {
                        "matrix_path": str(matrix_path),
                        "rhs_path": str(rhs_path),
                    },
                    "generation": {
                        "normalize": "none",
                        "shuffle": True,
                        "seed": 42,
                        "strategy": [{"name": "random", "samples": 20}],
                    },
                    "output": {"data_dir": str(data_dir / "processed")},
                },
                f,
            )

        output_root = data_dir / "output"
        job_config_path = jobs_dir / "mlflow_test_job.toml"
        _write_training_job_config(
            job_config_path,
            experiment_name="MLflowTestJob",
            model_class="ScaleEquivariantFFNN",
            enable_tracking=True,
        )

        experiment = load_assignment(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=data_config_path.stem),
        )

        assert experiment.settings.tracking.backend == "mlflow"
        assert (
            experiment.settings.training.trainer.default_root_dir == experiment.workspace.root_dir
        )
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()
        assert output_root in experiment.workspace.root_dir.parents

    def test_case_mlflow_topology_still_controls_output_root(
        self,
        training_setup: dict,
        neuralls_settings,
    ) -> None:
        """Case-config MLflow topology still controls derived output placement."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        tmp_path = training_setup["tmp_path"]
        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        jobs_dir = training_setup["jobs_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        data_config_path = datasets_dir / "injection_test.toml"
        with open(data_config_path, "wb") as f:
            tomli_w.dump(
                {
                    "id": "injection_test",
                    "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
                    "generation": {
                        "normalize": "none",
                        "strategy": [{"name": "random", "samples": 10}],
                    },
                    "output": {"data_dir": str(data_dir / "processed")},
                },
                f,
            )

        job_config_path = jobs_dir / "injection_job.toml"
        _write_training_job_config(
            job_config_path,
            experiment_name="InjectionJob",
            model_class="TestModel",
        )

        experiments_config_path = tmp_path / "experiments.toml"
        custom_output_root = data_dir / "custom_output"
        with open(experiments_config_path, "wb") as f:
            tomli_w.dump(
                {
                    "mlflow": {
                        "tracking_uri": f"sqlite:///{(custom_output_root / 'mlruns' / 'mlflow.db').as_posix()}",
                    },
                    "names": {
                        "training": "Train",
                        "comparison": "Comparisons",
                    },
                    "datasets": [{"id": "injection_test", "path": "datasets/injection_test.toml"}],
                    "jobs": [{"id": "injection_job", "path": "jobs/injection_job.toml"}],
                    "assignments": [
                        {
                            "id": "ignored",
                            "dataset": "injection_test",
                            "job": "injection_job",
                        }
                    ],
                },
                f,
            )

        experiment = load_assignment(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            neuralls_settings=neuralls_settings,
            case_config_path=experiments_config_path,
            identity=AssignmentIdentity(dataset_registry_id=data_config_path.stem),
        )

        assert experiment.settings.tracking.backend == "mlflow"
        assert custom_output_root in experiment.workspace.root_dir.parents


def _write_fit_job_config(path: Path, *, experiment_name: str) -> None:
    """Write a minimal lower-case one-shot fit job (`run.type = "fit"`, no `[training]`)."""
    job_config = {
        "run": {"type": "fit", "seed": 42},
        "experiment": {"name": experiment_name},
        "model": {
            "name": "PODCoarseningStrategy",
            "module_path": "torchalg.preconditioners.implementations.pod",
            "rank": 8,
        },
        "data": {
            "name": "FlexibleDataset",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "module": {"name": "ArrayDataModule"},
        },
        "tracking": {"backend": "mlflow"},
    }
    with open(path, "wb") as f:
        tomli_w.dump(job_config, f)


def test_load_assignment_accepts_fit_job_without_trainer_section(
    training_setup: dict,
    neuralls_settings,
) -> None:
    """A `run.type = "fit"` assignment (no `[training]` section) loads without
    crashing on `patch_runtime_workspace`'s trainer guard clause.

    Regression guard for `FitJobConfig`/`TrainableJobConfig` wiring
    (`_job_types.py`, `patch_runtime_workspace_for_job`,
    `assembler.py::_require_trainable_job`): before that wiring, this raised
    either `ConfigValidationError` (job loader's stale FitJobConfig blocklist)
    or `ValueError` (`patch_runtime_workspace`'s "Training jobs require
    [training].trainer" guard, unconditionally applied to every job kind).
    """
    from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

    data_dir = training_setup["data_dir"]
    datasets_dir = training_setup["datasets_dir"]
    jobs_dir = training_setup["jobs_dir"]
    matrix_path = training_setup["matrix_path"]
    rhs_path = training_setup["rhs_path"]

    data_config_path = datasets_dir / "fit_test_data.toml"
    with open(data_config_path, "wb") as f:
        tomli_w.dump(
            {
                "id": "fit_test_data",
                "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
                "generation": {
                    "normalize": "none",
                    "shuffle": True,
                    "seed": 42,
                    "strategy": [{"name": "random", "samples": 20}],
                },
                "output": {"data_dir": str(data_dir / "processed")},
            },
            f,
        )

    output_root = data_dir / "fit_output"
    job_config_path = jobs_dir / "fit_test_job.toml"
    _write_fit_job_config(job_config_path, experiment_name="FitTestJob")

    experiment = load_assignment(
        job_config_path=job_config_path,
        data_config_path=data_config_path,
        neuralls_settings=neuralls_settings,
        output_root=output_root,
        identity=AssignmentIdentity(dataset_registry_id=data_config_path.stem),
    )

    assert experiment.settings.training is None
    assert experiment.settings.tracking.backend == "mlflow"
