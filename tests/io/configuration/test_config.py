"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment
from neuralls.platform.config.models.workspace import AssignmentWorkspace, RunnableAssignment

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
)


@pytest.fixture
def temp_config_structure(tmp_path: Path) -> Path:
    """Create a temporary directory structure for config files."""
    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs" / "datasets").mkdir()
    (project_root / "configs" / "jobs").mkdir()
    (project_root / "configs" / "models").mkdir()
    (project_root / "configs" / "profiles").mkdir()

    with open(project_root / "configs" / "models" / "exp1_model.toml", "w") as f:
        f.write(
            '[model]\nname = "TestModel"\n\n[data]\nname = "FlexibleDataset"\nbatch_size = 2\n\n[data.module]\nname = "ArrayDataModule"'
        )

    with open(project_root / "configs" / "profiles" / "training.toml", "w") as f:
        f.write("[training.trainer]\nmax_epochs = 1")

    with open(project_root / "configs" / "jobs" / "exp1_job.toml", "w") as f:
        f.write(
            '[run]\ntype = "train"\nseed = 42\nmodel = "../models/exp1_model.toml"\ndata = "../models/exp1_model.toml"\ntraining = "../profiles/training.toml"\n\n[experiment]\nname = "exp1_job"'
        )

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('id="exp1_data_dataset"\n[source]\nmatrix_path="matrix.txt"\n')
    with open(project_root / "configs" / "datasets" / "default_data.toml", "w") as f:
        f.write('id="default_data_dataset"\n[source]\nmatrix_path="matrix.txt"\n')

    return project_root


def test_load_experiment_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_assignment correctly loads a single experiment."""
    monkeypatch.chdir(temp_config_structure)

    model_path = temp_config_structure / "configs" / "jobs" / "exp1_job.toml"
    data_path = temp_config_structure / "configs" / "datasets" / "exp1_data.toml"

    experiment = load_assignment(
        job_config_path=model_path,
        data_config_path=data_path,
        output_root=temp_config_structure / "output",
        identity=AssignmentIdentity(dataset_registry_id=data_path.stem),
    )

    assert isinstance(experiment, RunnableAssignment)
    assert isinstance(experiment.workspace, AssignmentWorkspace)
    assert experiment.spec.assignment_id == "exp1_job"
    assert experiment.spec.job_config_path == model_path
    assert experiment.spec.data_config_path == data_path

    assert (
        Path(experiment.settings.training.trainer.default_root_dir) == experiment.workspace.root_dir
    )


def test_load_experiment_missing_data_config(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that missing data config raises TypeError (required parameter)."""
    monkeypatch.chdir(temp_config_structure)
    model_path = temp_config_structure / "configs" / "jobs" / "exp1_job.toml"

    with pytest.raises(ValueError, match="data_config_path is required"):
        load_assignment(job_config_path=model_path)
