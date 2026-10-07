"""Tests for lower-case dataset runtime helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.config.job_config import TrainingJobConfig

from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.models.dataset import (
    create_features_from_array,
    create_matrix_feature,
    create_targets_from_array,
    with_dataset_arrays,
)


def _build_training_job(
    tmp_path: Path, *, dataset_name: str = "FlexibleDataset"
) -> TrainingJobConfig:
    trainer_root = tmp_path / "trainer-root"
    trainer_root.mkdir(parents=True, exist_ok=True)
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 42},
            "model": {"name": "LinearModel", "module_path": "dlkit.nn"},
            "data": {
                "name": dataset_name,
                "features": [],
                "targets": [],
                "module": {"name": "ArrayDataModule"},
            },
            "training": {
                "trainer": {
                    "max_epochs": 1,
                    "default_root_dir": str(trainer_root),
                }
            },
        }
    )


@pytest.fixture
def sample_rhs_array() -> np.ndarray:
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def sample_solutions_array() -> np.ndarray:
    return np.array([[0.5, 0.3], [0.2, 0.8]], dtype=np.float64)


@pytest.fixture
def sample_matrix_array() -> np.ndarray:
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


class TestCreateFeaturesFromArray:
    def test_create_features_default_name(self, sample_rhs_array: np.ndarray) -> None:
        contract = default_training_dataset_contract()
        features = create_features_from_array(sample_rhs_array, name=contract.primary_input_name)
        assert isinstance(features, tuple)
        assert len(features) == 1
        assert features[0].name == "x"

    def test_create_features_custom_name(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array, name="custom_feature")
        assert len(features) == 1
        assert features[0].name == "custom_feature"

    def test_create_features_model_input_is_true(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array, name="x")
        assert features[0].model_input is True


class TestCreateMatrixFeature:
    def test_create_matrix_feature(self, sample_matrix_array: np.ndarray) -> None:
        feature = create_matrix_feature(sample_matrix_array)
        assert feature.name == "matrix"
        assert feature.model_input is False

    def test_create_matrix_feature_custom_name(self, sample_matrix_array: np.ndarray) -> None:
        feature = create_matrix_feature(sample_matrix_array, name="stiffness")
        assert feature.name == "stiffness"
        assert feature.model_input is False


class TestCreateTargetsFromArray:
    def test_create_targets_default_name(self, sample_solutions_array: np.ndarray) -> None:
        contract = default_training_dataset_contract()
        targets = create_targets_from_array(sample_solutions_array, name=contract.target_name)
        assert isinstance(targets, tuple)
        assert len(targets) == 1
        assert targets[0].name == "y"

    def test_create_targets_write_defaults_to_false(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        targets = create_targets_from_array(sample_solutions_array, name="y")
        assert targets[0].write is False


class TestWithDatasetArrays:
    def test_with_dataset_arrays_standard_dataset(
        self,
        tmp_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
    ) -> None:
        settings = _build_training_job(tmp_path)
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            settings,
            sample_rhs_array,
            sample_solutions_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )
        assert updated.data.features[0].name == "x"
        assert updated.data.targets[0].name == "y"

    def test_with_dataset_arrays_graph_dataset_includes_matrix(
        self,
        tmp_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        sample_matrix_array: np.ndarray,
    ) -> None:
        settings = _build_training_job(tmp_path, dataset_name="GraphDataset")
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            settings,
            sample_rhs_array,
            sample_solutions_array,
            sample_matrix_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )
        assert any(f.name == "matrix" for f in updated.data.features)

    def test_with_dataset_arrays_keeps_original_settings_immutable(
        self,
        tmp_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
    ) -> None:
        settings = _build_training_job(tmp_path)
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            settings,
            sample_rhs_array,
            sample_solutions_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )

        assert updated is not settings
        assert settings.data.features == ()
        assert settings.data.targets == ()
        assert updated.data.targets is not None
