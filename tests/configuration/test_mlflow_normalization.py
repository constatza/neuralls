"""Unit tests for MLflow hard-cut validation and runtime helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from neuralls.platform.config.model_mlflow import normalize_model_mlflow
from neuralls.platform.config.resolution import (
    build_mlflow_environment,
    build_sqlite_tracking_uri,
    derive_output_root_from_tracking_uri,
    derive_sqlite_artifacts_destination,
    normalize_artifacts_destination,
    normalize_tracking_uri,
)
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from tests.path_assertions import assert_local_path_eq

WORKSPACE = Path("/workspace")


def test_normalize_model_mlflow_rejects_legacy_nested_sections() -> None:
    config_path = WORKSPACE / "model.toml"
    raw = {"MLFLOW": {"enabled": True, "client": {"run_name": "legacy"}}}

    with pytest.raises(ValueError):
        normalize_model_mlflow(raw, config_path)


def test_normalize_model_mlflow_rejects_tracking_uri_in_model_config() -> None:
    config_path = WORKSPACE / "model.toml"
    raw = {
        "MLFLOW": {
            "enabled": True,
            "tracking_uri": build_sqlite_tracking_uri(WORKSPACE / "mlruns" / "mlflow.db"),
        }
    }

    with pytest.raises(ValueError):
        normalize_model_mlflow(raw, config_path)


def test_normalize_model_mlflow_rejects_artifacts_destination_in_model_config() -> None:
    config_path = WORKSPACE / "model.toml"
    raw = {"MLFLOW": {"enabled": True, "artifacts_destination": "mlartifacts"}}

    with pytest.raises(ValueError):
        normalize_model_mlflow(raw, config_path)


def test_normalize_tracking_uri_resolves_relative_sqlite_path() -> None:
    config_path = WORKSPACE / "configs" / "model.toml"

    normalized = normalize_tracking_uri(
        "sqlite:///../output/mlruns/mlflow.db",
        config_path=config_path,
    )

    expected_db = (config_path.parent / "output/mlruns/mlflow.db").resolve()
    assert normalized == build_sqlite_tracking_uri(expected_db)


def test_normalize_tracking_uri_rejects_invalid_scheme() -> None:
    with pytest.raises(ValueError):
        normalize_tracking_uri("file:///tests/fixtures/mlflow.db")


def test_normalize_artifacts_destination_resolves_relative_path() -> None:
    config_path = WORKSPACE / "configs" / "experiments.toml"

    normalized = normalize_artifacts_destination(
        "../output/mlartifacts",
        config_path=config_path,
    )

    expected = (config_path.parent / "../output/mlartifacts").resolve()
    assert normalized == str(expected)


def test_build_mlflow_environment_derives_sqlite_artifact_uri() -> None:
    tracking_uri = build_sqlite_tracking_uri(WORKSPACE / "mlruns" / "mlflow.db")

    env = build_mlflow_environment(tracking_uri=tracking_uri)

    assert env["MLFLOW_TRACKING_URI"] == tracking_uri
    assert_local_path_eq(env["MLFLOW_ARTIFACT_URI"], (WORKSPACE / "mlartifacts").resolve())
    assert env["MLFLOW_ARTIFACT_URI"] == derive_sqlite_artifacts_destination(tracking_uri)


def test_build_mlflow_environment_keeps_remote_tracking_without_artifact_uri() -> None:
    tracking_uri = "https://tests.invalid/mlflow"
    env = build_mlflow_environment(tracking_uri=tracking_uri)

    assert env == {"MLFLOW_TRACKING_URI": tracking_uri}


def test_derive_output_root_from_tracking_uri_uses_repo_output_root() -> None:
    tracking_uri = build_sqlite_tracking_uri(WORKSPACE / "output" / "mlruns" / "mlflow.db")

    output_root = derive_output_root_from_tracking_uri(tracking_uri)

    assert output_root == (WORKSPACE / "output").resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-shaped path test")
def test_normalize_tracking_uri_accepts_windows_drive_letter_path(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "experiments.toml"
    config_path.parent.mkdir(parents=True)

    normalized = normalize_tracking_uri(
        "sqlite:///C:/neuralls/output/mlruns/mlflow.db",
        config_path=config_path,
    )

    assert normalized == "sqlite:///C:/neuralls/output/mlruns/mlflow.db"


def test_scoped_mlflow_environment_restores_previous_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    old_tracking = "https://previous.invalid/mlflow"
    new_tracking = "https://current.invalid/mlflow"
    artifact_uri = str(tmp_path / "mlartifacts")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", old_tracking)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    with scoped_mlflow_environment(
        {
            "MLFLOW_TRACKING_URI": new_tracking,
            "MLFLOW_ARTIFACT_URI": artifact_uri,
        }
    ):
        assert os.environ["MLFLOW_TRACKING_URI"] == new_tracking
        assert os.environ["MLFLOW_ARTIFACT_URI"] == artifact_uri

    assert os.environ["MLFLOW_TRACKING_URI"] == old_tracking
    assert "MLFLOW_ARTIFACT_URI" not in os.environ
