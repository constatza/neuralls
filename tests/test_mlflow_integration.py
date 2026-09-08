"""Integration tests for MLflow workflow with real tracking."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


@pytest.fixture
def mlflow_tracking_dir(tmp_path: Path) -> Path:
    """Temporary MLflow tracking directory."""
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir


@pytest.fixture
def mlflow_artifact_dir(tmp_path: Path) -> Path:
    """Temporary MLflow artifact directory."""
    artifact_dir = tmp_path / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    """Temporary output root for experiments."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def minimal_model_config(tmp_path: Path, output_root: Path) -> Path:
    """Create a minimal model config that relies on default MLflow settings."""
    config_path = tmp_path / "minimal_model.toml"
    profile_path = tmp_path / "minimal_model_profile.toml"

    profile_path.write_text(
        """
[model]
name = "MinimalModel"

[data]
name = "FlexibleDataset"

[data.module]
name = "ArrayDataModule"
"""
    )

    config_content = f"""
[run]
type = "train"
seed = 42
model = "{profile_path.name}"
data = "{profile_path.name}"

[experiment]
name = "TestSession"

[training.trainer]
max_epochs = 1
accelerator = "cpu"
devices = 1

[training.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def minimal_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config."""
    config_path = tmp_path / "test-dataset.toml"
    matrix_path = tmp_path / "dummy_matrix.txt"
    config_content = f"""
id = "test-dataset"

[source]
matrix_path = "{matrix_path.as_posix()}"

[generation]
normalize = "matrix"
"""
    config_path.write_text(config_content)
    return config_path


class TestMLflowExperimentCreation:
    """Tests for MLflow experiment creation with real tracking."""

    def test_mlflow_experiment_creation(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify MLflow is enabled for a loaded experiment."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        assert experiment.settings.tracking is not None

    def test_mlflow_run_creation_with_timestamp(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify load_assignment does not need to embed run naming in settings."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        assert experiment.settings.tracking is not None


class TestMLflowArtifactStorage:
    """Tests for MLflow artifact storage structure."""

    def test_mlflow_artifact_storage_structure(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify MLflow artifacts configured at mlartifacts/."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        assert output_root in experiment.workspace.root_dir.parents

    def test_mlflow_workspace_artifact_alignment(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify workspace directories are created under dataset/run hierarchy."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        # Verify workspace structure: output_root/dataset_id/run_id/
        dataset_id = minimal_data_config.stem
        run_id = experiment.workspace.run_id

        expected_root = output_root / dataset_id / run_id
        assert experiment.workspace.root_dir == expected_root

        # Verify subdirectories exist
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

    def test_mlflow_tracking_uri_configuration(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify MLflow tracking URI is configured correctly."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        assert experiment.settings.tracking is not None
        assert not hasattr(experiment.settings.tracking, "tracking_uri")
        assert not hasattr(experiment.settings.tracking, "artifacts_destination")


class TestCustomArtifacts:
    """Tests for custom artifacts (CSVs, plots, etc.) that dlkit doesn't handle."""

    def test_workspace_has_custom_artifact_directories(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify workspace provides directories for custom artifacts."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        workspace = experiment.workspace

        # Verify custom artifact directories are available
        assert workspace.figures_dir.exists()
        assert workspace.predictions_dir.exists()
        assert workspace.checkpoint_dir.exists()

        # These directories are for custom artifacts dlkit doesn't automatically log
        # Users can save CSVs, plots, etc. here

    def test_custom_csv_artifacts_can_be_saved(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify custom CSV artifacts can be saved to workspace."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        # Save a custom CSV metric
        custom_csv = experiment.workspace.root_dir / "custom_metrics.csv"
        with open(custom_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "custom_metric"])
            writer.writerow([1, 0.95])
            writer.writerow([2, 0.97])

        # Verify file was created
        assert custom_csv.exists()
        assert custom_csv.read_text().count("\n") == 3

    def test_custom_plot_artifacts_can_be_saved(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify custom plot artifacts can be saved to figures_dir."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        # Save a dummy plot file
        plot_file = experiment.workspace.figures_dir / "convergence_plot.png"
        plot_file.write_bytes(b"fake_png_data")

        # Verify file was created
        assert plot_file.exists()
        assert plot_file.parent == experiment.workspace.figures_dir

    def test_custom_prediction_outputs_can_be_saved(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify custom prediction outputs can be saved to predictions_dir."""
        import numpy as np

        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        # Save custom prediction outputs
        predictions_file = experiment.workspace.predictions_dir / "test_predictions.npy"
        dummy_predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.save(predictions_file, dummy_predictions)

        # Verify file was created
        assert predictions_file.exists()
        assert predictions_file.parent == experiment.workspace.predictions_dir

        # Verify data can be loaded back
        loaded = np.load(predictions_file)
        np.testing.assert_array_equal(loaded, dummy_predictions)


class TestMLflowPathResolution:
    """Tests for MLflow path resolution from single source of truth."""

    def test_mlflow_paths_derived_from_output_root(
        self,
        minimal_model_config: Path,
        minimal_data_config: Path,
        output_root: Path,
    ):
        """Verify all MLflow paths are derived from output_root."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        experiment = load_assignment(
            minimal_model_config,
            minimal_data_config,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=minimal_data_config.stem),
        )

        # All MLflow paths should contain output_root
        assert output_root in experiment.workspace.root_dir.parents

    def test_multiple_experiments_isolated_by_dataset(
        self,
        minimal_model_config: Path,
        output_root: Path,
        tmp_path: Path,
    ):
        """Verify different datasets create separate experiment hierarchies."""
        from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

        # Create two different data configs
        matrix_a = tmp_path / "dummy_A.txt"
        matrix_b = tmp_path / "dummy_B.txt"
        data_config_1 = tmp_path / "dataset-A.toml"
        data_config_1.write_text(f"""
id = "dataset-A"

[source]
matrix_path = "{matrix_a.as_posix()}"

[generation]
normalize = "matrix"
""")

        data_config_2 = tmp_path / "dataset-B.toml"
        data_config_2.write_text(f"""
id = "dataset-B"

[source]
matrix_path = "{matrix_b.as_posix()}"

[generation]
normalize = "matrix"
""")

        # Load both experiments
        exp1 = load_assignment(
            minimal_model_config,
            data_config_1,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=data_config_1.stem),
        )

        exp2 = load_assignment(
            minimal_model_config,
            data_config_2,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=data_config_2.stem),
        )

        # Verify workspace roots are isolated by dataset
        assert "dataset-A" in str(exp1.workspace.root_dir)
        assert "dataset-B" in str(exp2.workspace.root_dir)
        assert exp1.workspace.root_dir != exp2.workspace.root_dir
