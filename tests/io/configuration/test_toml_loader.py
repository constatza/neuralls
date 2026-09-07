"""Tests for typed TOML loader functions."""

from __future__ import annotations

import tomllib
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from neuralls.platform.config.loaders import (
    load_case_config,
    load_comparison_config,
    load_data_config,
    load_raw_toml,
)
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.settings import NeurallsSettings


def test_load_data_config_resolves_paths(
    minimal_data_config_toml: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Dataset loader expands NEURALLS placeholders into absolute paths."""
    config = load_data_config(minimal_data_config_toml, neuralls_settings)
    assert config.id == "test-dataset"
    assert config.source.matrix_path == str(
        (neuralls_settings.processed_dir / "matrix.mtx").resolve()
    )


def test_load_data_config_requires_id(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Dataset configs without an id fail validation."""
    config_file = tmp_path / "data.toml"
    config_file.write_text(
        """
[source]
matrix_path = "matrix.txt"
"""
    )
    with pytest.raises(ValidationError, match="id"):
        load_data_config(config_file, neuralls_settings)


def test_load_data_config_rejects_graph_cg_placeholder(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Stale GRAPH_CG placeholders are rejected with migration text."""
    config_file = tmp_path / "data.toml"
    config_file.write_text(
        """
id = "test"

[source]
matrix_path = "${GRAPH_CG_RAW_DIR}/matrix.txt"
"""
    )
    with pytest.raises(ValueError, match="NEURALLS"):
        load_data_config(config_file, neuralls_settings)


def test_load_comparison_config_resolves_paths(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Comparison loader expands matrix and rhs paths."""
    config_file = tmp_path / "comparison.toml"
    config_file.write_text(
        """
[general]

[general.params]
rtol = 1e-6

[general.data]
matrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.npy"
rhs_path = "${NEURALLS_PROCESSED_DIR}/rhs.npy"

[[preconditioners]]
name = "jacobi"
type = "jacobi"
"""
    )
    config = load_comparison_config(config_file, neuralls_settings)
    assert (
        config.general.data.matrix_path
        == (neuralls_settings.processed_dir / "matrix.npy").resolve()
    )
    assert config.general.data.rhs_path == (neuralls_settings.processed_dir / "rhs.npy").resolve()


def test_load_case_config_returns_typed_model(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Case loader returns CaseConfig directly."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
id = "dataset"
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert isinstance(config, CaseConfig)
    assert config.datasets[0].path == dataset_cfg.resolve()


def test_load_case_config_fills_missing_dataset_id_from_dataset_config(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A [[datasets]] entry without an id defaults to the dataset config's own id."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset-own-id"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert config.datasets[0].id == "dataset-own-id"


def test_load_case_config_explicit_dataset_id_wins(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """An explicit [[datasets]] id is never overridden by the dataset config's own id."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset-own-id"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
id = "case-local-alias"
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert config.datasets[0].id == "case-local-alias"


def test_load_case_config_requires_dataset_id_when_neither_has_one(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A missing id on both the case entry and the dataset config is a clear error."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text('[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n')
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
path = "{dataset_cfg.as_posix()}"
"""
    )
    with pytest.raises(ValueError, match="has no 'id'"):
        load_case_config(config_file, neuralls_settings)


def _write_dataset_config(path: Path, dataset_id: str) -> None:
    path.write_text(
        f'id = "{dataset_id}"\n[source]\nmatrix_path = "${{NEURALLS_PROCESSED_DIR}}/matrix.mtx"\n'
    )


def test_load_case_config_expands_dataset_sweeps(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """[[dataset_sweeps]] expands into real [[datasets]] entries with ids read from disk."""
    _write_dataset_config(tmp_path / "gaussian-cg50-1000.toml", "gaussian-cg50-family-1000")
    _write_dataset_config(tmp_path / "gaussian-cg50-2000.toml", "gaussian-cg50-family-2000")
    config_file = tmp_path / "case.toml"
    config_file.write_text(
        """
[[dataset_sweeps]]
label = "cg50-samples"
path_template = "gaussian-cg50-{value}.toml"
values = [1000, 2000]
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert [d.id for d in config.datasets] == [
        "gaussian-cg50-family-1000",
        "gaussian-cg50-family-2000",
    ]


def test_load_case_config_expands_assignment_sweeps(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """[[assignment_sweeps]] pairs every swept dataset id with the given job."""
    _write_dataset_config(tmp_path / "gaussian-cg50-1000.toml", "gaussian-cg50-family-1000")
    _write_dataset_config(tmp_path / "gaussian-cg50-2000.toml", "gaussian-cg50-family-2000")
    config_file = tmp_path / "case.toml"
    config_file.write_text(
        """
[[dataset_sweeps]]
label = "cg50-samples"
path_template = "gaussian-cg50-{value}.toml"
values = [1000, 2000]

[[jobs]]
id = "pod-2g_cg-50"
path = "jobs/dummy-job.toml"

[[assignment_sweeps]]
dataset_sweep = "cg50-samples"
job = "pod-2g_cg-50"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert [a.dataset_id for a in config.assignments] == [
        "gaussian-cg50-family-1000",
        "gaussian-cg50-family-2000",
    ]
    assert all(a.job_id == "pod-2g_cg-50" for a in config.assignments)


def test_load_case_config_rejects_unknown_dataset_sweep_label(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_file = tmp_path / "case.toml"
    config_file.write_text(
        """
[[jobs]]
id = "pod-2g_cg-50"
path = "jobs/dummy-job.toml"

[[assignment_sweeps]]
dataset_sweep = "typo"
job = "pod-2g_cg-50"
"""
    )
    with pytest.raises(ValueError, match="typo"):
        load_case_config(config_file, neuralls_settings)


def test_load_case_config_rejects_dataset_sweep_missing_value_placeholder(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_file = tmp_path / "case.toml"
    config_file.write_text(
        """
[[dataset_sweeps]]
label = "cg50-samples"
path_template = "gaussian-cg50-fixed.toml"
values = [1000, 2000]
"""
    )
    with pytest.raises(ValueError, match="path_template"):
        load_case_config(config_file, neuralls_settings)


def test_load_case_config_dataset_sweeps_duplicate_id_still_rejected(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A sweep whose values resolve to a shared id is still caught by the existing dedupe check."""
    _write_dataset_config(tmp_path / "gaussian-cg50-1000.toml", "same-id")
    _write_dataset_config(tmp_path / "gaussian-cg50-2000.toml", "same-id")
    config_file = tmp_path / "case.toml"
    config_file.write_text(
        """
[[dataset_sweeps]]
label = "cg50-samples"
path_template = "gaussian-cg50-{value}.toml"
values = [1000, 2000]
"""
    )
    with pytest.raises(ValidationError, match="Duplicate dataset registry ids"):
        load_case_config(config_file, neuralls_settings)


def test_load_case_config_sweep_expansion_matches_hand_written_equivalent(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A sweep-expressed case loads to the same datasets/assignments as the hand-written form."""
    _write_dataset_config(tmp_path / "gaussian-cg50-1000.toml", "gaussian-cg50-family-1000")
    _write_dataset_config(tmp_path / "gaussian-cg50-2000.toml", "gaussian-cg50-family-2000")

    swept_config = tmp_path / "swept.toml"
    swept_config.write_text(
        """
[[jobs]]
id = "pod-2g_cg-50"
path = "jobs/dummy-job.toml"

[[dataset_sweeps]]
label = "cg50-samples"
path_template = "gaussian-cg50-{value}.toml"
values = [1000, 2000]

[[assignment_sweeps]]
dataset_sweep = "cg50-samples"
job = "pod-2g_cg-50"
"""
    )
    hand_written_config = tmp_path / "hand_written.toml"
    hand_written_config.write_text(
        """
[[jobs]]
id = "pod-2g_cg-50"
path = "jobs/dummy-job.toml"

[[datasets]]
path = "gaussian-cg50-1000.toml"

[[datasets]]
path = "gaussian-cg50-2000.toml"

[[assignments]]
dataset = "gaussian-cg50-family-1000"
job = "pod-2g_cg-50"

[[assignments]]
dataset = "gaussian-cg50-family-2000"
job = "pod-2g_cg-50"
"""
    )
    swept = load_case_config(swept_config, neuralls_settings)
    hand_written = load_case_config(hand_written_config, neuralls_settings)
    assert [d.id for d in swept.datasets] == [d.id for d in hand_written.datasets]
    assert [(a.dataset_id, a.job_id) for a in swept.assignments] == [
        (a.dataset_id, a.job_id) for a in hand_written.assignments
    ]


def test_load_raw_toml_success(tmp_path: Path) -> None:
    """Raw TOML loading returns a plain dict."""
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[section]
key = "value"
number = 42
"""
    )
    data = load_raw_toml(config_file)
    assert isinstance(data, dict)
    assert data["section"]["key"] == "value"
    assert data["section"]["number"] == 42


def test_load_raw_toml_invalid_syntax(tmp_path: Path) -> None:
    """Invalid TOML raises the stdlib decode error."""
    config_file = tmp_path / "test.toml"
    config_file.write_text("invalid [ syntax")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_raw_toml(config_file)


def test_load_raw_toml_rejects_unescaped_windows_path(tmp_path: Path) -> None:
    """Raw Windows backslashes inside TOML basic strings trigger stdlib parsing errors."""
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        r"""
[source]
matrix_path = "C:\Users\runneradmin\AppData\Local\Temp\matrix.txt"
"""
    )
    with pytest.raises(tomllib.TOMLDecodeError, match="Invalid hex value"):
        load_raw_toml(config_file)


def test_load_raw_toml_accepts_normalized_windows_path(tmp_path: Path) -> None:
    """POSIX-normalized Windows paths remain valid TOML literals everywhere."""
    config_file = tmp_path / "test.toml"
    matrix_path = PureWindowsPath(r"C:\Users\runneradmin\AppData\Local\Temp\matrix.txt")
    config_file.write_text(f'[source]\nmatrix_path = "{matrix_path.as_posix()}"\n')
    data = load_raw_toml(config_file)
    assert data["source"]["matrix_path"] == matrix_path.as_posix()
