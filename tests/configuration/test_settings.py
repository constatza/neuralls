"""Tests for NeurallsSettings and explicit env-file loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.settings import (
    ENV_FILE_ENV_VAR,
    NeurallsSettings,
    get_settings,
)

WORKSPACE = Path("/workspace")


def test_settings_loads_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct BaseSettings construction reads NEURALLS_* env vars."""
    processed = WORKSPACE / "processed"
    output = WORKSPACE / "output"
    for key, value in {
        "NEURALLS_PROCESSED_DIR": processed,
        "NEURALLS_OUTPUT_DIR": output,
    }.items():
        monkeypatch.setenv(key, str(value))
    settings = NeurallsSettings(_env_file=[])
    assert settings.processed_dir == processed.resolve()
    assert settings.output_dir == output.resolve()


def test_settings_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required env vars fail fast."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(ENV_FILE_ENV_VAR, raising=False)
    with pytest.raises(ValidationError):
        get_settings()


def test_settings_derived_paths(neuralls_settings: NeurallsSettings) -> None:
    """Derived directories live under output_dir."""
    assert neuralls_settings.figures_dir == neuralls_settings.output_dir / "figures"
    assert neuralls_settings.mlruns_dir == neuralls_settings.output_dir / "mlruns"
    assert neuralls_settings.mlartifacts_dir == neuralls_settings.output_dir / "mlartifacts"


def test_settings_mlflow_uri_uses_posix(neuralls_settings: NeurallsSettings) -> None:
    """The sqlite tracking URI uses forward slashes."""
    assert "\\" not in neuralls_settings.mlflow_tracking_uri
    assert neuralls_settings.mlflow_tracking_uri.startswith("sqlite:///")


def test_settings_paths_resolved_absolute(neuralls_settings: NeurallsSettings) -> None:
    """All root path fields are absolute after validation."""
    assert neuralls_settings.processed_dir.is_absolute()
    assert neuralls_settings.output_dir.is_absolute()


def test_mlflow_artifact_location_uses_str_not_posix(neuralls_settings: NeurallsSettings) -> None:
    """mlflow_artifact_location uses str(), not as_posix(), so UNC paths on Windows keep backslashes."""
    assert neuralls_settings.mlflow_artifact_location == str(neuralls_settings.mlartifacts_dir)
