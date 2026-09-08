"""Integration tests for lower-case DLKit job loading and experiment assembly."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.common.errors import ConfigValidationError
from dlkit.infrastructure.config.job_config import SearchJobConfig, TrainingJobConfig

from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment
from neuralls.platform.config.dlkit_bridge import load_job_config
from neuralls.platform.config.models.workspace import (
    AssignmentSpec,
    AssignmentWorkspace,
    RunnableAssignment,
)


def _write_model_profile(
    path: Path,
    *,
    model_name: str,
) -> Path:
    path.write_text(
        f"""
[model]
name = "{model_name}"

[data]
name = "FlexibleDataset"
batch_size = 64

[data.module]
name = "ArrayDataModule"
"""
    )
    return path


def _write_job_config(
    path: Path,
    *,
    model_profile_path: Path,
    run_type: str = "train",
    experiment_name: str | None = "test-model",
) -> Path:
    experiment_block = f'[experiment]\nname = "{experiment_name}"\n\n' if experiment_name else ""
    path.write_text(
        f"""
[run]
type = "{run_type}"
seed = 42
model = "{model_profile_path.relative_to(path.parent).as_posix()}"
data = "{model_profile_path.relative_to(path.parent).as_posix()}"

{experiment_block}[training.trainer]
max_epochs = 1

[training.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
"""
    )
    return path


@pytest.fixture
def sample_model_config(tmp_path: Path) -> Path:
    """Create a minimal lower-case DLKit job config TOML."""
    model_profile = _write_model_profile(
        tmp_path / "test-model-profile.toml", model_name="TestModel"
    )
    return _write_job_config(tmp_path / "model.toml", model_profile_path=model_profile)


@pytest.fixture
def sample_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config TOML."""
    config_path = tmp_path / "data.toml"
    matrix_path = tmp_path / "test_matrix.txt"
    config_path.write_text(
        f"""
id = "test-data"

[source]
matrix_path = "{matrix_path.as_posix()}"

[generation]
normalize = "matrix"
"""
    )
    return config_path


class TestJobLoading:
    """Tests for the DLKit-backed job adapter."""

    def test_load_job_config_loads_search_jobs(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Search jobs load as SearchJobConfig."""
        model_profile = _write_model_profile(
            tmp_path / "search-profile.toml", model_name="TestModel"
        )
        config_path = _write_job_config(
            tmp_path / "search-model.toml",
            model_profile_path=model_profile,
            run_type="search",
            experiment_name="search-model",
        )
        config_path.write_text(
            config_path.read_text()
            + """

[search]
space."training.optimizer.default_optimizer.lr" = { type = "float", low = 1e-4, high = 1e-2 }
"""
        )

        settings = load_job_config(config_path, neuralls_settings)
        assert isinstance(settings, SearchJobConfig)

    def test_load_job_config_keeps_training_jobs_as_training(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Train jobs remain TrainingJobConfig."""
        model_profile = _write_model_profile(
            tmp_path / "train-profile.toml", model_name="TestModel"
        )
        config_path = _write_job_config(
            tmp_path / "train-model.toml",
            model_profile_path=model_profile,
            run_type="train",
            experiment_name="train-model",
        )

        settings = load_job_config(config_path, neuralls_settings)
        assert isinstance(settings, TrainingJobConfig)

    def test_load_job_config_rejects_legacy_uppercase_jobs(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Uppercase legacy manifests fail immediately via DLKit's own [run] validation."""
        config_path = tmp_path / "legacy-model.toml"
        config_path.write_text(
            """
[SESSION]
name = "legacy-model"
seed = 42
workflow = "train"

[MODEL]
name = "TestModel"

[TRAINING.trainer]
max_epochs = 1
"""
        )

        with pytest.raises(ConfigValidationError, match="No run.type found"):
            load_job_config(config_path, neuralls_settings)


class TestLoadExperiment:
    """Integration tests for load_assignment."""

    def test_load_experiment_success(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Experiment loading returns the expected wrapper types."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_assignment(
            job_config_path=sample_model_config,
            data_config_path=sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert isinstance(experiment, RunnableAssignment)
        assert isinstance(experiment.spec, AssignmentSpec)
        assert isinstance(experiment.workspace, AssignmentWorkspace)

    def test_experiment_spec_fields(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Experiment spec stores the resolved ids and paths."""
        experiment = load_assignment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=tmp_path,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.spec.assignment_id == "test-model"
        assert experiment.spec.job_config_path == sample_model_config
        assert experiment.spec.data_config_path == sample_data_config
        assert experiment.spec.checkpoint_path is None

    def test_workspace_fields(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Workspace paths are rooted under the dataset config's own id, not the
        case-registry alias — deliberately passing a different alias here
        (the filename stem, "data") than the dataset config's own id
        ("test-data") to guard against the two drifting apart on disk."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_assignment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.workspace.dataset_id == "test-data"
        assert experiment.workspace.run_id == "test-model"
        assert experiment.workspace.root_dir.parent == output_root / "test-data"
        assert experiment.workspace.root_dir.name == "test-model"
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

    def test_settings_integration(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Training settings are patched with workspace and tracking defaults."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_assignment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert (
            Path(experiment.settings.training.trainer.default_root_dir)
            == experiment.workspace.root_dir
        )
        assert experiment.settings.tracking.backend == "mlflow"

    def test_load_with_model_name_from_model_section(
        self,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Model name is used when experiment.name is omitted."""
        model_profile = _write_model_profile(
            tmp_path / "model-only-profile.toml", model_name="OnlyModelName"
        )
        job = _write_job_config(
            tmp_path / "model_only.toml",
            model_profile_path=model_profile,
            experiment_name=None,
        )

        experiment = load_assignment(
            job,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=tmp_path,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.spec.assignment_id == "OnlyModelName"
        assert experiment.workspace.run_id == "OnlyModelName"
