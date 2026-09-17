"""Tests for the case-centric configuration loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neuralls.composition.assignments.assembler import load_assignment_batch
from neuralls.platform.config.models.workspace import AssignmentWorkspace

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
)


@pytest.fixture
def temp_config_structure(tmp_path: Path) -> Path:
    """Create a temporary directory structure for NEW config format."""
    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs" / "datasets").mkdir()
    (project_root / "configs" / "jobs").mkdir()
    (project_root / "configs" / "models").mkdir()
    (project_root / "configs" / "profiles").mkdir()
    matrix_path = project_root / "data" / "matrix.txt"
    matrix_path.parent.mkdir()
    matrix_path.write_text("1.0\n")
    matrix2_path = project_root / "data" / "matrix2.txt"
    matrix2_path.write_text("2.0\n")

    # Case config uses [[assignments]] registry entries.
    with open(project_root / "configs" / "experiments.toml", "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "exp1_data"\n')
        f.write('path = "datasets/exp1_data.toml"\n\n')
        f.write("[[datasets]]\n")
        f.write('id = "exp2_data"\n')
        f.write('path = "datasets/exp2_data.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write('id = "exp1_job"\n')
        f.write('path = "jobs/exp1_job.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write('id = "exp2_job"\n')
        f.write('path = "jobs/exp2_job.toml"\n\n')
        f.write("# Assignment 1: Full config with explicit checkpoint\n")
        f.write("[[assignments]]\n")
        f.write('id = "exp1"\n')
        f.write('dataset = "exp1_data"\n')
        f.write('job = "exp1_job"\n')
        f.write(
            f'checkpoint_path = "{(project_root / "checkpoints" / "exp1.ckpt").as_posix()}"\n\n'
        )
        f.write("# Assignment 2: Config without checkpoint (will warn)\n")
        f.write("[[assignments]]\n")
        f.write('id = "exp2"\n')
        f.write('dataset = "exp2_data"\n')
        f.write('job = "exp2_job"\n')

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('id = "exp1_data"\n\n')
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix_path.as_posix()}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{(project_root / "data" / "processed").as_posix()}"\n')

    with open(project_root / "configs" / "datasets" / "exp2_data.toml", "w") as f:
        f.write('id = "exp2_data"\n\n')
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix2_path.as_posix()}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{(project_root / "data" / "processed").as_posix()}"\n')

    # Job configs
    with open(project_root / "configs" / "models" / "exp1_model.toml", "w") as f:
        f.write("[model]\n")
        f.write('name = "TestModel"\n')
        f.write("\n")
        f.write("[data]\n")
        f.write('name = "FlexibleDataset"\n\n')
        f.write("[data.module]\n")
        f.write('name = "ArrayDataModule"\n')

    with open(project_root / "configs" / "models" / "exp2_model.toml", "w") as f:
        f.write("[model]\n")
        f.write('name = "TestModel2"\n')
        f.write('module_path = "dlkit.domain.nn.graph"\n\n')
        f.write("[data]\n")
        f.write('name = "FlexibleDataset"\n\n')
        f.write("[data.module]\n")
        f.write('name = "ArrayDataModule"\n')

    with open(project_root / "configs" / "profiles" / "training.toml", "w") as f:
        f.write("[training.trainer]\n")
        f.write("max_epochs = 1\n")

    with open(project_root / "configs" / "jobs" / "exp1_job.toml", "w") as f:
        f.write("[run]\n")
        f.write('type = "train"\n')
        f.write("seed = 42\n")
        f.write('precision = "64"\n')
        f.write('model = "../models/exp1_model.toml"\n\n')
        f.write('data = "../models/exp1_model.toml"\n')
        f.write('training = "../profiles/training.toml"\n\n')
        f.write("[experiment]\n")
        f.write('name = "exp1_job"\n')

    with open(project_root / "configs" / "jobs" / "exp2_job.toml", "w") as f:
        f.write("[run]\n")
        f.write('type = "train"\n')
        f.write("seed = 42\n")
        f.write('precision = "64"\n')
        f.write('model = "../models/exp2_model.toml"\n\n')
        f.write('data = "../models/exp2_model.toml"\n')
        f.write('training = "../profiles/training.toml"\n\n')
        f.write("[experiment]\n")
        f.write('name = "exp2_job"\n')

    return project_root


def test_load_assignments_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_assignment_batch correctly loads assignments with NEW format."""
    monkeypatch.chdir(temp_config_structure)

    batch = load_assignment_batch(
        case_config_path=temp_config_structure / "configs" / "experiments.toml",
    )

    assert len(batch.assignments) == 2

    exp1 = batch.assignments[0]
    exp2 = batch.assignments[1]

    # --- Check Assignment 1 (with checkpoint) ---
    # spec.assignment_id preserves the master registry assignment id.
    assert exp1.spec.assignment_id == "exp1"
    assert exp1.settings is not None
    assert isinstance(exp1.workspace, AssignmentWorkspace)

    # Check paths resolve to shared directories
    assert exp1.spec.job_config_path.name == "exp1_job.toml"
    assert "jobs" in str(exp1.spec.job_config_path)
    assert exp1.spec.data_config_path.name == "exp1_data.toml"
    assert "datasets" in str(exp1.spec.data_config_path)

    # Check checkpoint path
    assert exp1.spec.checkpoint_path is not None
    assert "exp1.ckpt" in str(exp1.spec.checkpoint_path)

    # --- Check Assignment 2 (without checkpoint) ---
    assert exp2.spec.assignment_id == "exp2"

    # Check paths resolve to shared directories
    assert exp2.spec.job_config_path.name == "exp2_job.toml"
    assert "jobs" in str(exp2.spec.job_config_path)
    assert exp2.spec.data_config_path.name == "exp2_data.toml"
    assert "datasets" in str(exp2.spec.data_config_path)

    # No checkpoint for exp2
    assert exp2.spec.checkpoint_path is None


def test_load_assignments_missing_registry_id(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing registry ids fail before any guessed-path lookup."""
    monkeypatch.chdir(temp_config_structure)

    # Create experiments.toml with reference to non-existent dataset
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("[[jobs]]\n")
        f.write('id = "exp1_job"\n')
        f.write('path = "jobs/exp1_job.toml"\n\n')
        f.write("[[assignments]]\n")
        f.write('id = "exp_missing"\n')
        f.write('dataset = "missing_dataset"\n')
        f.write('job = "exp1_job"\n')

    with pytest.raises(
        ValueError, match="Assignment 'exp_missing' references dataset id 'missing_dataset'"
    ):
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_assignments_missing_generated_dataset_hints_at_expansion_script(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """A registered dataset path under `_generated/` that's missing hints at re-expanding it."""
    monkeypatch.chdir(temp_config_structure)

    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "exp1_data"\n')
        f.write('path = "datasets/_generated/gaussian-cg50-5000.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write('id = "exp1_job"\n')
        f.write('path = "jobs/exp1_job.toml"\n\n')
        f.write("[[assignments]]\n")
        f.write('id = "exp1"\n')
        f.write('dataset = "exp1_data"\n')
        f.write('job = "exp1_job"\n')

    with pytest.raises(FileNotFoundError, match="expand_dataset_sweep.py --all"):
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_assignments_missing_dataset_without_generated_segment_has_no_sweep_hint(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing dataset path outside `_generated/` skips the sweep-expansion hint."""
    monkeypatch.chdir(temp_config_structure)

    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "exp1_data"\n')
        f.write('path = "datasets/typo-dataset.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write('id = "exp1_job"\n')
        f.write('path = "jobs/exp1_job.toml"\n\n')
        f.write("[[assignments]]\n")
        f.write('id = "exp1"\n')
        f.write('dataset = "exp1_data"\n')
        f.write('job = "exp1_job"\n')

    with pytest.raises(FileNotFoundError) as exc_info:
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )
    assert "expand_dataset_sweep.py" not in str(exc_info.value)


def test_load_assignments_rejects_unknown_comparison_assignment_filter(
    temp_config_structure: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison assignments filter cannot reference ids missing from [[assignments]]."""
    monkeypatch.chdir(temp_config_structure)
    with open(temp_config_structure / "configs" / "experiments.toml", "a") as f:
        f.write("\n[[datasets]]\n")
        f.write('id = "solutions"\n')
        f.write('path = "datasets/exp1_data.toml"\n\n')
        f.write("[[datasets]]\n")
        f.write('id = "gaussian-rhs"\n')
        f.write('path = "datasets/exp2_data.toml"\n\n')
        f.write("[[comparisons]]\n")
        f.write('id = "gaussian"\n')
        f.write('matrix_dataset = "solutions"\n')
        f.write('rhs_source = { kind = "gaussian" }\n')
        f.write('assignments = ["missing-exp"]\n')

    with pytest.raises(
        ValueError,
        match="Comparison 'gaussian' assignments filter references unknown assignment ids: missing-exp",
    ):
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_assignments_no_assignments(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that ValueError is raised when no assignments defined."""
    monkeypatch.chdir(temp_config_structure)

    # Create empty experiments.toml
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("")

    with pytest.raises(ValueError, match="No assignments defined"):
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_case_config_rejects_legacy_models_table(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case configs must use [[jobs]] instead of the removed [[models]] table."""
    monkeypatch.chdir(temp_config_structure)

    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("[[models]]\n")
        f.write('id = "legacy-model"\n')
        f.write('path = "models/legacy.toml"\n')

    with pytest.raises(ValueError, match=r"\[\[jobs\]\]"):
        load_assignment_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )
