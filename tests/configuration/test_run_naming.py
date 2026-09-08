"""Unit tests for lower-case DLKit run naming logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment


def _write_model_profile(
    path: Path,
    *,
    model_name: str,
    dataset_name: str = "FlexibleDataset",
) -> Path:
    path.write_text(
        f"""
[model]
name = "{model_name}"

[data]
name = "{dataset_name}"

[data.module]
name = "ArrayDataModule"
"""
    )
    return path


def _write_job_config(
    path: Path,
    *,
    model_profile_path: Path,
    experiment_name: str | None = None,
) -> Path:
    experiment_block = f'\n[experiment]\nname = "{experiment_name}"\n' if experiment_name else ""
    path.write_text(
        f"""
[run]
type = "train"
seed = 42
model = "{model_profile_path.relative_to(path.parent).as_posix()}"
data = "{model_profile_path.relative_to(path.parent).as_posix()}"

[training.trainer]
max_epochs = 1

[training.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3{experiment_block}
"""
    )
    return path


@pytest.fixture
def job_config_with_experiment_name(tmp_path: Path) -> Path:
    """Create a job config whose experiment.name should drive run naming."""
    model_profile = _write_model_profile(tmp_path / "ffnn-profile.toml", model_name="FFNNModel")
    return _write_job_config(
        tmp_path / "job_with_experiment.toml",
        model_profile_path=model_profile,
        experiment_name="MyCustomExperiment",
    )


@pytest.fixture
def job_config_without_experiment_name(tmp_path: Path) -> Path:
    """Create a job config that falls back to model.name."""
    model_profile = _write_model_profile(
        tmp_path / "linear-profile.toml",
        model_name="NormScaledLinearFFNN",
    )
    return _write_job_config(
        tmp_path / "job_without_experiment.toml",
        model_profile_path=model_profile,
        experiment_name=None,
    )


@pytest.fixture
def sample_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config TOML."""
    config_path = tmp_path / "test-dataset.toml"
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


class TestRunNaming:
    """Tests for stable lower-case job naming."""

    def test_run_id_falls_back_to_model_name(
        self,
        job_config_without_experiment_name: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        experiment = load_assignment(
            job_config_without_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.workspace.run_id == "NormScaledLinearFFNN"

    def test_run_id_uses_experiment_name_when_set(
        self,
        job_config_with_experiment_name: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        experiment = load_assignment(
            job_config_with_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.workspace.run_id == "MyCustomExperiment"

    def test_run_id_is_stable_across_loads(
        self,
        job_config_without_experiment_name: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        exp1 = load_assignment(
            job_config_without_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output1",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )
        exp2 = load_assignment(
            job_config_without_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output2",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert exp1.workspace.run_id == exp2.workspace.run_id == "NormScaledLinearFFNN"

    def test_experiment_spec_id_matches_run_id(
        self,
        job_config_without_experiment_name: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        experiment = load_assignment(
            job_config_without_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.spec.assignment_id == "NormScaledLinearFFNN"
        assert experiment.workspace.run_id == "NormScaledLinearFFNN"

    def test_workspace_directories_use_run_id(
        self,
        job_config_without_experiment_name: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        output_root = tmp_path / "output"
        experiment = load_assignment(
            job_config_without_experiment_name,
            sample_data_config,
            neuralls_settings,
            output_root=output_root,
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        run_id = experiment.workspace.run_id
        expected_root = output_root / "test-data" / run_id
        assert experiment.workspace.root_dir == expected_root
        assert str(run_id) in str(experiment.workspace.checkpoint_dir)


class TestRunNamingEdgeCases:
    """Edge cases and error scenarios for run naming."""

    def test_missing_model_name_falls_back_to_job_stem(
        self,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        model_profile = tmp_path / "bad-profile.toml"
        model_profile.write_text(
            """
[model]
name = ""

[data]
name = "FlexibleDataset"

[data.module]
name = "ArrayDataModule"
"""
        )
        bad_config = _write_job_config(
            tmp_path / "bad-model.toml",
            model_profile_path=model_profile,
            experiment_name=None,
        )

        experiment = load_assignment(
            bad_config,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.workspace.run_id == "bad-model"

    def test_run_id_handles_special_characters_in_experiment_name(
        self,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        model_profile = _write_model_profile(
            tmp_path / "special-profile.toml", model_name="TestModel"
        )
        special_config = _write_job_config(
            tmp_path / "special-model.toml",
            model_profile_path=model_profile,
            experiment_name="Model_release-alpha",
        )

        experiment = load_assignment(
            special_config,
            sample_data_config,
            neuralls_settings,
            output_root=tmp_path / "output",
            identity=AssignmentIdentity(dataset_registry_id=sample_data_config.stem),
        )

        assert experiment.workspace.run_id == "Model_release-alpha"
