"""Tests for explicit env-file loading (the env file is read from disk)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.settings import ENV_FILE_ENV_VAR, get_settings


def test_settings_env_var_overrides_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process env overrides explicit env-file values."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        f"NEURALLS_PROCESSED_DIR={tmp_path / 'custom/processed'}\nNEURALLS_OUTPUT_DIR={tmp_path / 'custom/output'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", str(tmp_path / "override-output"))

    settings = get_settings(env_file)

    assert settings.processed_dir == (tmp_path / "custom/processed").resolve()
    assert settings.output_dir == (tmp_path / "override-output").resolve()


def test_settings_custom_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env_file is the only dotenv source when provided."""
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(ENV_FILE_ENV_VAR, raising=False)
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        f"NEURALLS_PROCESSED_DIR={tmp_path / 'custom/processed'}\nNEURALLS_OUTPUT_DIR={tmp_path / 'custom/output'}\n",
        encoding="utf-8",
    )

    settings = get_settings(env_file)

    assert settings.processed_dir == (tmp_path / "custom/processed").resolve()
    assert settings.output_dir == (tmp_path / "custom/output").resolve()


def test_settings_uses_neuralls_env_file_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEURALLS_ENV_FILE is consulted only when env_file is absent."""
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        f"NEURALLS_PROCESSED_DIR={tmp_path / 'custom/processed'}\nNEURALLS_OUTPUT_DIR={tmp_path / 'custom/output'}\n",
        encoding="utf-8",
    )
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_FILE_ENV_VAR, str(env_file))

    settings = get_settings()

    assert settings.processed_dir == (tmp_path / "custom/processed").resolve()
    assert settings.output_dir == (tmp_path / "custom/output").resolve()
