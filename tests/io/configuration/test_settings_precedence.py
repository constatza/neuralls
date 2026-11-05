"""Integration tests for explicit settings and profile/env precedence."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import neuralls.platform.config.profile as profile_module
from neuralls.platform.config.profile import ProfileConfig, save_profile
from neuralls.platform.config.settings import get_settings, load_case_settings


def _write_case_config(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _configure_profile_path(monkeypatch: pytest.MonkeyPatch, config_file: Path) -> None:
    monkeypatch.setattr(profile_module, "USER_CONFIG_DIR", config_file.parent)
    monkeypatch.setattr(profile_module, "USER_CONFIG_FILE", config_file)


def test_precedence_env_var_over_env_file_over_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Process env beats env file, which beats active profile values."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    _write_case_config(case_config)
    profile_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, profile_file)
    save_profile(
        "default",
        ProfileConfig(
            raw_dir=tmp_path / "profile-raw",
            processed_dir=tmp_path / "profile-processed",
            output_dir=tmp_path / "profile-output",
        ),
        _config_file=profile_file,
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "NEURALLS_PROCESSED_DIR=/env-file/processed\nNEURALLS_OUTPUT_DIR=/env-file/output\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", str(tmp_path / "env-output"))

    settings = load_case_settings(case_config, env_file)

    assert settings.processed_dir == Path("/env-file/processed").resolve()
    assert settings.output_dir == (tmp_path / "env-output").resolve()


def test_precedence_env_file_over_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit env file beats profile values when process env is absent."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    _write_case_config(case_config)
    profile_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, profile_file)
    save_profile(
        "default",
        ProfileConfig(
            raw_dir=tmp_path / "profile-raw",
            processed_dir=tmp_path / "profile-processed",
            output_dir=tmp_path / "profile-output",
        ),
        _config_file=profile_file,
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "NEURALLS_PROCESSED_DIR=/env-file/processed\nNEURALLS_OUTPUT_DIR=/env-file/output\n",
        encoding="utf-8",
    )

    settings = load_case_settings(case_config, env_file)

    assert settings.processed_dir == Path("/env-file/processed").resolve()
    assert settings.output_dir == Path("/env-file/output").resolve()


def test_no_env_files_required_env_vars_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Env-only loading still works when no env file is provided."""
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    for key, value in {
        "NEURALLS_PROCESSED_DIR": processed,
        "NEURALLS_OUTPUT_DIR": output,
    }.items():
        monkeypatch.setenv(key, str(value))

    settings = get_settings()

    assert settings.processed_dir == processed.resolve()
    assert settings.output_dir == output.resolve()


def test_get_settings_does_not_scan_cwd_env_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ambient cwd .env files are ignored by design."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "NEURALLS_PROCESSED_DIR=/cwd/processed\nNEURALLS_OUTPUT_DIR=/cwd/output\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    with pytest.raises(ValidationError):
        get_settings()
