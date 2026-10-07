"""Tests for lower-case training runtime patch architecture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dlkit.infrastructure.config.data_entries import DataRole, NpyEntry, ValueEntry, ZarrEntry
from dlkit.infrastructure.config.job_config import FitJobConfig, TrainingJobConfig

from neuralls.composition.assignments._dataset_assembly import (
    _create_feature_entries,
    _create_target_entries,
    validate_runtime_dataset_contract,
)
from neuralls.composition.assignments._settings_pipeline import _configure_training_pipeline
from neuralls.composition.assignments._training_artifacts import (
    _extract_evaluation_arrays,
    _normalize_training_numpy_payload,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    RuntimeDatasetContract,
    default_training_dataset_contract,
)
from neuralls.composition.assignments.runtime_dataset_patcher import patch_runtime_dataset
from neuralls.composition.assignments.runtime_tracking_patcher import patch_training_tracking
from neuralls.composition.assignments.runtime_workspace_patcher import (
    patch_dataloader_runtime,
    patch_runtime_workspace,
    patch_runtime_workspace_for_job,
)
from neuralls.platform.config.dataset_entries import apply_placeholder_metadata
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.storage.training_artifacts import (
    NpyArraySource,
    TrainingArrays,
    ZarrArraySource,
    matrix_zarr_path,
)
from neuralls.shared.types import MatrixFormat


def _build_training_job(tmp_path: Path) -> TrainingJobConfig:
    trainer_root = tmp_path / "original-root"
    trainer_root.mkdir(parents=True, exist_ok=True)
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 42},
            "model": {"name": "LinearModel", "module_path": "dlkit.nn"},
            "data": {
                "name": "FlexibleDataset",
                "batch_size": 8,
                "num_workers": 2,
                "pin_memory": True,
                "features": [],
                "targets": [],
                "module": {"name": "ArrayDataModule"},
            },
            "training": {
                "trainer": {
                    "max_epochs": 3,
                    "default_root_dir": str(trainer_root),
                    "callbacks": [],
                }
            },
            "tracking": {},
        }
    )


@pytest.fixture
def sample_arrays(tmp_path: Path) -> TrainingArrays:
    rng = np.random.default_rng(seed=42)
    rhs = rng.random((10, 5)).astype(np.float64)
    solutions = rng.random((10, 5)).astype(np.float64)
    rhs_path = tmp_path / "rhs.npy"
    solutions_path = tmp_path / "solutions.npy"
    np.save(rhs_path, rhs)
    np.save(solutions_path, solutions)
    matrix_zarr = tmp_path / "matrix.zarr"
    matrix_zarr.mkdir()
    (matrix_zarr / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    return TrainingArrays(
        rhs_source=NpyArraySource(path=rhs_path),
        solutions_source=NpyArraySource(path=solutions_path),
        matrix_source=ZarrArraySource(path=matrix_zarr),
        sample_count=10,
    )


@pytest.fixture
def training_settings(tmp_path: Path) -> TrainingJobConfig:
    return _build_training_job(tmp_path)


def _build_fit_job() -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit", "seed": 42},
            "model": {
                "name": "PODCoarseningStrategy",
                "module_path": "torchalg.preconditioners.implementations.pod",
                "rank": 8,
            },
            "data": {
                "name": "FlexibleDataset",
                "batch_size": 8,
                "num_workers": 2,
                "pin_memory": True,
                "features": [],
                "targets": [],
                "module": {"name": "ArrayDataModule"},
            },
            "tracking": {},
        }
    )


@pytest.fixture
def fit_settings() -> FitJobConfig:
    return _build_fit_job()


@pytest.fixture
def workspace(tmp_path: Path) -> AssignmentWorkspace:
    root = tmp_path / "output" / "dataset" / "run"
    return AssignmentWorkspace(
        dataset_id="dataset",
        run_id="run",
        root_dir=root,
        data_dir=root / "data",
    )


def test_create_feature_configs_returns_rhs_and_matrix(sample_arrays: TrainingArrays) -> None:
    contract = default_training_dataset_contract()
    features = _create_feature_entries(
        sample_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=True
    )

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    rhs_feature = next(f for f in features if f.name == "x")
    assert isinstance(rhs_feature, NpyEntry)
    assert rhs_feature.path == sample_arrays.rhs_source.path
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, ZarrEntry)
    assert matrix_feature.model_input is False
    assert matrix_feature.path == matrix_zarr_path(sample_arrays)


def test_create_feature_configs_omits_matrix_when_not_declared(
    sample_arrays: TrainingArrays,
) -> None:
    """DLKit never needs the matrix for its own pipeline — it must not be attached
    unless a model explicitly declared it via ``[[data.features]]``."""
    contract = default_training_dataset_contract()
    features = _create_feature_entries(
        sample_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=False
    )

    assert [f.name for f in features] == ["x"]


def test_create_feature_configs_rejects_matrix_on_csr_storage(
    sample_arrays: TrainingArrays,
) -> None:
    """A declared matrix feature against CSR storage must fail fast with a clear
    neuralls-level error, not a cryptic AttributeError from inside DLKit's reader."""
    contract = default_training_dataset_contract()
    csr_arrays = replace(sample_arrays, matrix_format=MatrixFormat.CSR)

    with pytest.raises(ValueError, match="sparse_io backend"):
        _create_feature_entries(
            csr_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=True
        )


def test_create_target_configs_returns_canonical_supervised_target(
    sample_arrays: TrainingArrays,
) -> None:
    contract = default_training_dataset_contract()
    targets = _create_target_entries(sample_arrays.solutions_source, contract)

    assert len(targets) == 1
    assert targets[0].name == "y"
    assert isinstance(targets[0], NpyEntry)
    assert targets[0].data_role == DataRole.TARGET
    assert targets[0].path == sample_arrays.solutions_source.path


def test_patch_runtime_dataset_returns_new_settings(
    training_settings: TrainingJobConfig,
) -> None:
    contract = default_training_dataset_contract()
    features = [ValueEntry(name="x", value=np.zeros((1, 1), dtype=np.float64))]
    targets = [
        ValueEntry(
            name="y",
            value=np.zeros((1, 1), dtype=np.float64),
            data_role=DataRole.TARGET,
        )
    ]
    updated = patch_runtime_dataset(
        training_settings,
        features=features,
        targets=targets,
        contract=contract,
    )
    assert updated is not training_settings
    assert [entry.name for entry in updated.data.features] == ["x"]
    assert [entry.name for entry in updated.data.targets] == ["y"]
    assert training_settings.data.features == ()
    assert training_settings.data.targets == ()


def test_validate_runtime_dataset_contract_rejects_duplicate_target_names(
    training_settings: TrainingJobConfig,
) -> None:
    duplicate_targets = training_settings.patch(
        {
            "data": {
                "targets": [
                    ValueEntry(
                        name="y",
                        value=np.zeros((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    ),
                    ValueEntry(
                        name="y",
                        value=np.ones((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    ),
                ]
            }
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="Duplicate data target entry names"):
        validate_runtime_dataset_contract(duplicate_targets, contract)


def test_validate_runtime_dataset_contract_rejects_duplicate_feature_names(
    training_settings: TrainingJobConfig,
) -> None:
    duplicate_features = training_settings.patch(
        {
            "data": {
                "features": [
                    ValueEntry(name="x", value=np.zeros((1, 1), dtype=np.float64)),
                    ValueEntry(name="x", value=np.ones((1, 1), dtype=np.float64)),
                ]
            }
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="Duplicate data feature entry names"):
        validate_runtime_dataset_contract(duplicate_features, contract)


def test_validate_runtime_dataset_contract_rejects_matrix_feature_on_fit_job(
    fit_settings: FitJobConfig,
) -> None:
    """A run.type='fit' job is matrix-free by design; declaring 'matrix' must fail fast."""
    declares_matrix = fit_settings.patch(
        {"data": {"features": [ValueEntry(name="matrix", value=np.zeros((1, 1)))]}}
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="run.type='fit' job cannot declare"):
        validate_runtime_dataset_contract(declares_matrix, contract)


def test_patch_runtime_workspace_returns_new_settings(
    training_settings: TrainingJobConfig,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "new-root"
    output_dir.mkdir()

    updated = patch_runtime_workspace(training_settings, output_dir=output_dir)

    assert updated is not training_settings
    assert updated.training.trainer.default_root_dir == output_dir
    assert updated.training.trainer.callbacks[-1].name == "RetainedCheckpointCopy"


def test_patch_runtime_workspace_for_job_patches_train_like_jobs(
    training_settings: TrainingJobConfig,
    tmp_path: Path,
) -> None:
    """`TrainLikeJobConfig` (has a trainer) is patched exactly like the raw function."""
    output_dir = tmp_path / "new-root"
    output_dir.mkdir()

    updated = patch_runtime_workspace_for_job(training_settings, output_dir=output_dir)

    assert updated is not training_settings
    assert updated.training.trainer.default_root_dir == output_dir


def test_patch_runtime_workspace_for_job_is_noop_for_fit_jobs(
    fit_settings: FitJobConfig,
    tmp_path: Path,
) -> None:
    """`FitJobConfig` has no `training` section — patching it is a no-op, not a crash."""
    output_dir = tmp_path / "new-root"
    output_dir.mkdir()

    updated = patch_runtime_workspace_for_job(fit_settings, output_dir=output_dir)

    assert updated is fit_settings
    assert updated.training is None


def test_patch_dataloader_runtime_clamps_workers_for_zarr(
    training_settings: TrainingJobConfig,
) -> None:
    updated = patch_dataloader_runtime(training_settings, dataset_format="zarr")

    assert updated is not training_settings
    assert updated.data.num_workers == 0
    assert updated.data.pin_memory is True
    assert training_settings.data.num_workers == 2


def test_patch_dataloader_runtime_leaves_non_zarr_untouched(
    training_settings: TrainingJobConfig,
) -> None:
    updated = patch_dataloader_runtime(training_settings, dataset_format="hdf5")

    assert updated is training_settings
    assert updated.data.num_workers == 2


def test_patch_training_tracking_returns_new_settings(
    training_settings: TrainingJobConfig,
) -> None:
    updated = patch_training_tracking(training_settings, uri="http://localhost:5000")

    assert updated is not training_settings
    assert updated.tracking.backend == "mlflow"
    assert updated.tracking.uri == "http://localhost:5000"


def test_contract_override_drives_injection_and_validation(
    sample_arrays: TrainingArrays,
    training_settings: TrainingJobConfig,
    tmp_path: Path,
) -> None:
    contract = RuntimeDatasetContract(
        primary_input_name="lhs",
        matrix_input_name="matrix",
        target_name="rhs",
        prediction_name="rhs_pred",
        loss_target_key="targets.rhs",
    )
    features = _create_feature_entries(
        sample_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=True
    )
    targets = _create_target_entries(sample_arrays.solutions_source, contract)
    settings = training_settings.patch(
        {
            "data": {
                "features": [ValueEntry(name="lhs", value=np.zeros((1, 1), dtype=np.float64))],
                "targets": [
                    ValueEntry(
                        name="rhs",
                        value=np.zeros((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    )
                ],
            },
            "training": {"loss": {"target_key": "targets.rhs"}},
        }
    )

    validate_runtime_dataset_contract(settings, contract)
    runtime_root = tmp_path / "runtime-run"
    runtime_root.mkdir(parents=True, exist_ok=True)
    updated, _ = _configure_training_pipeline(
        settings,
        workspace=AssignmentWorkspace(
            dataset_id="dataset",
            run_id="run",
            root_dir=runtime_root,
            data_dir=runtime_root / "data",
        ),
        features=features,
        targets=targets,
        contract=contract,
        dataset_format="hdf5",
    )

    assert [feature.name for feature in updated.data.features] == ["lhs", "matrix"]
    assert [target.name for target in updated.data.targets] == ["rhs"]


@patch("neuralls.composition.assignments._training_artifacts.compute_diagnostics")
@patch("neuralls.composition.assignments._training_artifacts.write_diagnostics_figure")
@patch("neuralls.composition.assignments._training_artifacts.log_diagnostics_to_mlflow")
def test_log_training_evaluation_orchestration(
    mock_mlflow_log: MagicMock,
    mock_write: MagicMock,
    mock_compute: MagicMock,
    tmp_path: Path,
) -> None:
    from neuralls.composition.assignments._training_artifacts import _log_training_evaluation

    contract = default_training_dataset_contract()
    normalized_payload = {
        "predictions": {contract.prediction_name: np.zeros((10, 1))},
        "targets": {"y": np.zeros((10, 1))},
    }
    mock_compute.return_value = MagicMock()
    mock_write.return_value = Path("dummy.png")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id="run123",
        numpy_payload=normalized_payload,
        figures_dir=tmp_path / "figures",
        contract=contract,
    )

    mock_compute.assert_called_once()
    mock_write.assert_called_once()
    mock_mlflow_log.assert_called_once()


def test_extract_evaluation_arrays_with_canonical_prediction_key() -> None:
    contract = default_training_dataset_contract()
    selected = _extract_evaluation_arrays(
        {
            "predictions": {contract.prediction_name: np.array([[1.0], [2.0], [3.0]])},
            "targets": {contract.target_name: np.array([[1.1], [1.9], [3.2]])},
        },
        contract,
    )
    assert selected is not None
    y_pred, y_true = selected
    np.testing.assert_allclose(y_pred.ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_true.ravel(), [1.1, 1.9, 3.2])


def test_extract_evaluation_arrays_rejects_noncanonical_prediction_key() -> None:
    contract = default_training_dataset_contract()
    assert (
        _extract_evaluation_arrays(
            {
                "predictions": {"y_hat": np.array([[1.0], [2.0]])},
                "targets": {contract.target_name: np.array([[0.8], [2.2]])},
            },
            contract,
        )
        is None
    )


def test_extract_evaluation_arrays_returns_none_when_missing_keys() -> None:
    contract = default_training_dataset_contract()
    assert _extract_evaluation_arrays({"predictions": np.array([1.0, 2.0])}, contract) is None


def test_normalize_training_numpy_payload_maps_dlkit_output_once() -> None:
    contract = default_training_dataset_contract()
    payload = _normalize_training_numpy_payload(
        {
            "predictions": {"output": np.array([[1.0], [2.0]])},
            "targets": {"y": np.array([[0.5], [1.5]])},
        },
        contract,
    )
    assert payload is not None
    np.testing.assert_allclose(payload["predictions"][contract.prediction_name], [[1.0], [2.0]])


def test_apply_placeholder_metadata_preserves_existing_entry_metadata(tmp_path: Path) -> None:
    path = tmp_path / "x.npy"
    path.write_bytes(b"\x93NUMPY")
    entries = [NpyEntry(name="x", path=path)]
    placeholders = [ValueEntry(name="x", value=np.zeros((1, 1)), model_input=False)]
    updated = apply_placeholder_metadata(entries, tuple(placeholders))
    assert updated[0].name == "x"
    assert updated[0].model_input is False
