"""Tests for profile-based user config storage."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
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


def _set_root_env(monkeypatch: pytest.MonkeyPatch, raw: str, processed: str, output: str) -> None:
    monkeypatch.setenv("NEURALLS_RAW_DIR", raw)
    monkeypatch.setenv("NEURALLS_PROCESSED_DIR", processed)
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", output)


Roots = tuple[Path | None, Path | None, Path | None]
"""The (raw, processed, output) roots a source yields."""


@dataclass(frozen=True, slots=True)
class _RootsSource:
    """One way of turning raw/processed/output strings into roots: a stored profile or settings."""

    construct: Callable[[str, str, str], Roots]
    load: Callable[[Path, pytest.MonkeyPatch, str, str, str], Roots]
    processed_label: str


def _construct_profile(raw: str, processed: str, output: str) -> Roots:
    config = ProfileConfig(raw_dir=raw, processed_dir=processed, output_dir=output)
    return (config.raw_dir, config.processed_dir, config.output_dir)


def _construct_settings(raw: str, processed: str, output: str) -> Roots:
    settings = NeurallsSettings(
        _env_file=[], raw_dir=raw, processed_dir=processed, output_dir=output
    )
    return (settings.raw_dir, settings.processed_dir, settings.output_dir)


def _load_profile_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, processed: str, output: str
) -> Roots:
    config_file = tmp_path / "config.toml"
    _saved_profile(config_file, raw, processed, output)
    loaded = load_profile(_config_file=config_file)
    if loaded is None:
        raise LookupError(config_file)
    return (loaded.raw_dir, loaded.processed_dir, loaded.output_dir)


def _load_settings_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, processed: str, output: str
) -> Roots:
    _set_root_env(monkeypatch, raw, processed, output)
    settings = get_settings()
    return (settings.raw_dir, settings.processed_dir, settings.output_dir)


@pytest.fixture(
    params=[
        _RootsSource(_construct_profile, _load_profile_roots, "processed_dir"),
        _RootsSource(_construct_settings, _load_settings_roots, "NEURALLS_PROCESSED_DIR"),
    ],
    ids=["profile", "settings"],
)
def roots_source(request: pytest.FixtureRequest) -> _RootsSource:
    return request.param


def test_roots_are_kept_as_given(roots_source: _RootsSource) -> None:
    assert roots_source.construct("raw", "~/processed", "output") == (
        Path("raw"),
        Path("~/processed"),
        Path("output"),
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdlib expanduser prefers USERPROFILE over HOME on Windows",
)
def test_load_expands_tilde(
    roots_source: _RootsSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_dir: Path
) -> None:
    loaded = roots_source.load(tmp_path, monkeypatch, "~/raw", "~/processed", "~/output")

    assert loaded == (home_dir / "raw", home_dir / "processed", home_dir / "output")


def test_load_rejects_relative_root(
    roots_source: _RootsSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=roots_source.processed_label):
        roots_source.load(
            tmp_path, monkeypatch, str(tmp_path / "raw"), "processed", str(tmp_path / "output")
        )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_load_preserves_windows_mapped_drive_paths(
    roots_source: _RootsSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mapped-drive roots stay drive-letter-based on Windows."""
    loaded = roots_source.load(
        tmp_path,
        monkeypatch,
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )

    assert tuple(str(root) for root in loaded) == (
        r"M:\shared\neuralls\raw",
        r"M:\shared\neuralls\processed",
        r"M:\shared\neuralls\output",
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_load_rejects_windows_mapped_drive_paths_on_posix(
    roots_source: _RootsSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mapped-drive roots remain invalid on POSIX hosts."""
    with pytest.raises(ValueError):
        roots_source.load(
            tmp_path,
            monkeypatch,
            r"M:\shared\neuralls\raw",
            r"M:\shared\neuralls\processed",
            r"M:\shared\neuralls\output",
        )


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
