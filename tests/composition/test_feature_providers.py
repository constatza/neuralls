"""Tests for index-based extra feature mapping in _create_feature_configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dlkit.infrastructure.config.data_entries import ZarrEntry

from neuralls.composition.assignments._dataset_assembly import (
    _create_feature_entries,
    _extra_feature_names_from_settings,
    _wants_matrix_feature,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.storage.training_artifacts import (
    TrainingArrays,
    ZarrArraySource,
    parameters_zarr_paths,
)
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract():
    """Default runtime dataset contract."""
    return default_training_dataset_contract()


@pytest.fixture
def no_params_arrays(tmp_path: Path) -> TrainingArrays:
    """TrainingArrays with no parameters_zarr entries."""
    samples = 6
    for name in ("rhs.zarr", "solutions.zarr", "matrix.zarr"):
        store = tmp_path / name
        store.mkdir()
        (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    return TrainingArrays(
        rhs_source=ZarrArraySource(path=tmp_path / "rhs.zarr"),
        solutions_source=ZarrArraySource(path=tmp_path / "solutions.zarr"),
        matrix_source=ZarrArraySource(path=tmp_path / "matrix.zarr"),
        sample_count=samples,
        parameter_sources=(),
    )


@pytest.fixture
def one_param_arrays(tmp_path: Path) -> TrainingArrays:
    """TrainingArrays with one parameters_0.zarr of shape (6, 4)."""
    samples = 6
    for name in ("rhs.zarr", "solutions.zarr", "matrix.zarr"):
        store = tmp_path / name
        store.mkdir()
        (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    params_path = tmp_path / f"{PARAMETERS_ZARR_PREFIX}0.zarr"
    params_path.mkdir()
    (params_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    return TrainingArrays(
        rhs_source=ZarrArraySource(path=tmp_path / "rhs.zarr"),
        solutions_source=ZarrArraySource(path=tmp_path / "solutions.zarr"),
        matrix_source=ZarrArraySource(path=tmp_path / "matrix.zarr"),
        sample_count=samples,
        parameter_sources=(ZarrArraySource(path=params_path),),
    )


@pytest.fixture
def two_param_arrays(tmp_path: Path) -> TrainingArrays:
    """TrainingArrays with parameters_0.zarr and parameters_1.zarr."""
    samples = 6
    for name in ("rhs.zarr", "solutions.zarr", "matrix.zarr"):
        store = tmp_path / name
        store.mkdir()
        (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    paths = []
    for i in range(2):
        p = tmp_path / f"{PARAMETERS_ZARR_PREFIX}{i}.zarr"
        p.mkdir()
        (p / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
        paths.append(p)
    return TrainingArrays(
        rhs_source=ZarrArraySource(path=tmp_path / "rhs.zarr"),
        solutions_source=ZarrArraySource(path=tmp_path / "solutions.zarr"),
        matrix_source=ZarrArraySource(path=tmp_path / "matrix.zarr"),
        sample_count=samples,
        parameter_sources=tuple(ZarrArraySource(path=path) for path in paths),
    )


@pytest.fixture
def rhs_data() -> np.ndarray:
    """Small RHS array for feature config tests."""
    return np.zeros((6, 4), dtype=np.float64)


# ---------------------------------------------------------------------------
# _create_feature_configs — index-based ZarrEntry mapping
# ---------------------------------------------------------------------------


def test_no_extras_yields_x_and_matrix(
    no_params_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_entries(
        no_params_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=True
    )
    assert [e.name for e in entries] == ["x", "matrix"]


def test_matrix_omitted_when_not_declared(
    no_params_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_entries(
        no_params_arrays, contract, [], contract.primary_input_name, wants_matrix_feature=False
    )
    assert [e.name for e in entries] == ["x"]


def test_condition_extra_maps_to_parameters_0(
    one_param_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_entries(
        one_param_arrays,
        contract,
        ["condition"],
        contract.primary_input_name,
        wants_matrix_feature=True,
    )
    assert [e.name for e in entries] == ["x", "matrix", "condition"]
    condition_entry = entries[2]
    assert isinstance(condition_entry, ZarrEntry)
    assert condition_entry.path == parameters_zarr_paths(one_param_arrays)[0]


def test_two_extras_map_by_index(
    two_param_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_entries(
        two_param_arrays,
        contract,
        ["condition", "query"],
        contract.primary_input_name,
        wants_matrix_feature=True,
    )
    assert [e.name for e in entries] == ["x", "matrix", "condition", "query"]
    assert isinstance(entries[2], ZarrEntry)
    assert isinstance(entries[3], ZarrEntry)
    assert entries[2].path == parameters_zarr_paths(two_param_arrays)[0]
    assert entries[3].path == parameters_zarr_paths(two_param_arrays)[1]


def test_too_many_extras_raises(
    no_params_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    with pytest.raises(ValueError, match="extra features but dataset has"):
        _create_feature_entries(
            no_params_arrays,
            contract,
            ["condition"],
            contract.primary_input_name,
            wants_matrix_feature=True,
        )


def test_declaration_order_is_preserved(
    two_param_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries_ab = _create_feature_entries(
        two_param_arrays,
        contract,
        ["alpha", "beta"],
        contract.primary_input_name,
        wants_matrix_feature=True,
    )
    entries_ba = _create_feature_entries(
        two_param_arrays,
        contract,
        ["beta", "alpha"],
        contract.primary_input_name,
        wants_matrix_feature=True,
    )
    assert isinstance(entries_ab[2], ZarrEntry)
    assert isinstance(entries_ba[2], ZarrEntry)
    assert entries_ab[2].path == parameters_zarr_paths(two_param_arrays)[0]
    assert entries_ba[2].path == parameters_zarr_paths(two_param_arrays)[0]
    assert entries_ab[2].name == "alpha"
    assert entries_ba[2].name == "beta"


# ---------------------------------------------------------------------------
# _extra_feature_names_from_settings
# ---------------------------------------------------------------------------


def _make_minimal_settings(feature_names: list[str]):
    """Build a minimal lower-case DLKit TrainingJobConfig with placeholder feature entries."""
    from dlkit.infrastructure.config.data_entries import ValueEntry
    from dlkit.infrastructure.config.job_config import TrainingJobConfig

    features = tuple(ValueEntry(name=name) for name in feature_names)
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 42},
            "model": {"name": "FlexibleModel"},
            "data": {
                "name": "FlexibleDataset",
                "features": [f.model_dump() for f in features],
            },
            "training": {
                "trainer": {"max_epochs": 1},
                "loss": {
                    "name": "relative_vector_norm_loss",
                    "module_path": "dlkit.domain.losses",
                    "target_key": "targets.y",
                },
            },
        }
    )


def test_extra_names_returns_ordered_list(contract) -> None:
    settings = _make_minimal_settings(["x", "condition", "query"])
    extra = _extra_feature_names_from_settings(settings, contract)
    assert extra == ["condition", "query"]


def test_extra_names_empty_when_only_base(contract) -> None:
    settings = _make_minimal_settings(["x"])
    extra = _extra_feature_names_from_settings(settings, contract)
    assert extra == []


def test_wants_matrix_feature_false_when_not_declared(contract) -> None:
    settings = _make_minimal_settings(["x"])
    assert _wants_matrix_feature(settings, contract) is False


def test_wants_matrix_feature_true_when_declared(contract) -> None:
    settings = _make_minimal_settings(["x", "matrix"])
    assert _wants_matrix_feature(settings, contract) is True
