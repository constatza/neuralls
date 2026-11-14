"""Tests for profile-based user config storage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import neuralls.platform.config.profile as profile_module
from neuralls.platform.config.profile import (
    PROFILE_ENV_VAR,
    ProfileConfig,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from neuralls.platform.config.settings import NeurallsSettings, get_settings, load_case_settings


def _configure_profile_path(monkeypatch: pytest.MonkeyPatch, config_file: Path) -> None:
    monkeypatch.setattr(profile_module, "USER_CONFIG_DIR", config_file.parent)
    monkeypatch.setattr(profile_module, "USER_CONFIG_FILE", config_file)


def _profile(raw_root: Path, processed_root: Path, output_root: Path) -> ProfileConfig:
    return ProfileConfig(
        raw_dir=raw_root,
        processed_dir=processed_root,
        output_dir=output_root,
    )


def test_save_profile_creates_default_and_named_profiles(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    assert config_file.exists()
    assert list_profiles(_config_file=config_file) == ["default", "laptop"]


def test_load_profile_none_returns_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    expected = _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output")
    save_profile("default", expected, _config_file=config_file)

    assert load_profile(_config_file=config_file) == expected


def test_load_profile_named_returns_named_profile(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    expected = _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output")
    save_profile("laptop", expected, _config_file=config_file)

    assert load_profile("laptop", _config_file=config_file) == expected


def test_load_profile_missing_named_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )

    with pytest.raises(ValueError):
        load_profile("laptop", _config_file=config_file)


def test_load_profile_returns_none_when_config_file_missing(tmp_path: Path) -> None:
    assert load_profile(_config_file=tmp_path / "missing.toml") is None


def test_list_profiles_returns_default_then_named_profiles(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    assert list_profiles(_config_file=config_file) == ["default", "laptop"]


def test_delete_profile_removes_named_and_refuses_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    delete_profile("laptop", _config_file=config_file)
    assert list_profiles(_config_file=config_file) == ["default"]

    with pytest.raises(ValueError):
        delete_profile("default", _config_file=config_file)


def test_save_profile_duplicate_without_overwrite_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )

    with pytest.raises(FileExistsError):
        save_profile(
            "default",
            _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
            _config_file=config_file,
        )


def test_save_profile_with_overwrite_replaces_existing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    replacement = _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2")

    save_profile("default", replacement, overwrite=True, _config_file=config_file)

    assert load_profile(_config_file=config_file) == replacement


@pytest.fixture
def home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _saved_profile(config_file: Path, raw: str, processed: str, output: str) -> None:
    save_profile(
        "default",
        ProfileConfig(raw_dir=raw, processed_dir=processed, output_dir=output),
        _config_file=config_file,
    )


def _clear_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NEURALLS_RAW_DIR", "NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(name, raising=False)


def _set_root_env(monkeypatch: pytest.MonkeyPatch, raw: str, processed: str, output: str) -> None:
    monkeypatch.setenv("NEURALLS_RAW_DIR", raw)
    monkeypatch.setenv("NEURALLS_PROCESSED_DIR", processed)
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", output)


def test_profile_config_keeps_paths_as_given() -> None:
    config = ProfileConfig(raw_dir="raw", processed_dir="~/processed", output_dir="output")

    assert (config.raw_dir, config.processed_dir, config.output_dir) == (
        Path("raw"),
        Path("~/processed"),
        Path("output"),
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdlib expanduser prefers USERPROFILE over HOME on Windows",
)
def test_load_profile_expands_tilde(tmp_path: Path, home_dir: Path) -> None:
    config_file = tmp_path / "config.toml"
    _saved_profile(config_file, "~/raw", "~/processed", "~/output")

    loaded = load_profile(_config_file=config_file)

    assert loaded == ProfileConfig(
        raw_dir=home_dir / "raw",
        processed_dir=home_dir / "processed",
        output_dir=home_dir / "output",
    )


def test_load_profile_rejects_relative_root(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _saved_profile(config_file, str(tmp_path / "raw"), "processed", str(tmp_path / "output"))

    with pytest.raises(ValueError, match="processed_dir"):
        load_profile(_config_file=config_file)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_load_profile_preserves_windows_mapped_drive_paths(tmp_path: Path) -> None:
    """Mapped-drive roots should stay drive-letter-based on Windows."""
    config_file = tmp_path / "config.toml"
    _saved_profile(
        config_file,
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )

    loaded = load_profile(_config_file=config_file)

    assert loaded is not None
    assert str(loaded.raw_dir) == r"M:\shared\neuralls\raw"
    assert str(loaded.processed_dir) == r"M:\shared\neuralls\processed"
    assert str(loaded.output_dir) == r"M:\shared\neuralls\output"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_load_profile_rejects_windows_mapped_drive_paths_on_posix(tmp_path: Path) -> None:
    """Mapped-drive roots remain invalid on POSIX hosts."""
    config_file = tmp_path / "config.toml"
    _saved_profile(
        config_file,
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )

    with pytest.raises(ValueError):
        load_profile(_config_file=config_file)


def test_neuralls_settings_keeps_paths_as_given() -> None:
    settings = NeurallsSettings(_env_file=[], processed_dir="processed", output_dir="output")

    assert (settings.processed_dir, settings.output_dir) == (Path("processed"), Path("output"))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdlib expanduser prefers USERPROFILE over HOME on Windows",
)
def test_get_settings_expands_tilde(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
    _set_root_env(monkeypatch, "~/raw", "~/processed", "~/output")

    settings = get_settings()

    assert settings.raw_dir == home_dir / "raw"
    assert settings.processed_dir == home_dir / "processed"
    assert settings.output_dir == home_dir / "output"


def test_get_settings_rejects_relative_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_root_env(monkeypatch)
    monkeypatch.setenv("NEURALLS_PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", "output")

    with pytest.raises(ValueError, match="NEURALLS_OUTPUT_DIR"):
        get_settings()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_get_settings_preserves_windows_mapped_drive_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime settings should not collapse mapped drives into UNC paths on Windows."""
    _set_root_env(
        monkeypatch,
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )

    settings = get_settings()

    assert str(settings.raw_dir) == r"M:\shared\neuralls\raw"
    assert str(settings.processed_dir) == r"M:\shared\neuralls\processed"
    assert str(settings.output_dir) == r"M:\shared\neuralls\output"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_get_settings_rejects_windows_mapped_drive_paths_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime settings should reject Windows-only roots on POSIX hosts."""
    _set_root_env(
        monkeypatch,
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )

    with pytest.raises(ValueError):
        get_settings()


def test_profile_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    """processed_dir and output_dir are required; raw_dir is optional."""
    with pytest.raises(ValidationError):
        ProfileConfig.model_validate({"processed_dir": str(tmp_path / "processed")})
    with pytest.raises(ValidationError):
        ProfileConfig.model_validate({"output_dir": str(tmp_path / "output")})
    # raw_dir is optional — omitting it is valid
    config = ProfileConfig.model_validate(
        {"processed_dir": str(tmp_path / "processed"), "output_dir": str(tmp_path / "output")}
    )
    assert config.raw_dir is None


def test_profile_config_rejects_blank_or_whitespace_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProfileConfig.model_validate(
            {
                "raw_dir": " ",
                "processed_dir": str(tmp_path / "processed"),
                "output_dir": str(tmp_path / "output"),
            }
        )


def test_profile_precedence_profile_then_env_file_then_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    case_config.write_text("", encoding="utf-8")
    profile_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, profile_file)
    save_profile(
        "default",
        _profile(
            tmp_path / "profile-raw", tmp_path / "profile-processed", tmp_path / "profile-output"
        ),
        _config_file=profile_file,
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "\n".join(
            [
                f"NEURALLS_PROCESSED_DIR={tmp_path / 'env-processed'}",
                f"NEURALLS_OUTPUT_DIR={tmp_path / 'env-output'}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", str(tmp_path / "env-process-output"))

    settings = load_case_settings(env_file)

    assert settings.processed_dir == (tmp_path / "env-processed").resolve()
    assert settings.output_dir == (tmp_path / "env-process-output").resolve()


def test_neuralls_profile_env_var_selects_named_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, config_file)
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    expected = _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2")
    save_profile("laptop", expected, _config_file=config_file)
    monkeypatch.setenv(PROFILE_ENV_VAR, "laptop")

    assert load_profile(_config_file=config_file) == expected
