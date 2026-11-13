"""Tests for ConfigContext path expansion helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import neuralls.platform.config.context as context_module
from neuralls.platform.config.context import ConfigContext, expand_config_glob, expand_config_path
from neuralls.platform.config.settings import NeurallsSettings
from tests.os_paths import FAKE_ROOT

ROOT = FAKE_ROOT / "virtual"


def test_expand_neuralls_raw_dir_raises_when_not_configured(config_context) -> None:
    """`${NEURALLS_RAW_DIR}` raises when raw_dir is not set in settings."""
    with pytest.raises(ValueError):
        expand_config_path("${NEURALLS_RAW_DIR}/matrix.mtx", config_context)


def test_expand_neuralls_raw_dir_resolves_when_configured() -> None:
    """`${NEURALLS_RAW_DIR}` expands to the settings raw root when set."""
    raw = ROOT / "raw"
    settings = NeurallsSettings(
        _env_file=[],
        raw_dir=raw,
        processed_dir=ROOT / "processed",
        output_dir=ROOT / "output",
    )
    config_path = ROOT / "cfg.toml"
    from neuralls.platform.config.context import ConfigContext

    ctx = ConfigContext(config_path=config_path, settings=settings)
    result = expand_config_path("${NEURALLS_RAW_DIR}/matrix.mtx", ctx)
    assert result == str((raw / "matrix.mtx").resolve())


def test_expand_neuralls_output_dir(
    config_context,
    neuralls_settings: NeurallsSettings,
) -> None:
    """`${NEURALLS_OUTPUT_DIR}` expands to the settings output root."""
    value = expand_config_path("${NEURALLS_OUTPUT_DIR}/reports", config_context)
    assert value == str((neuralls_settings.output_dir / "reports").resolve())


def test_expand_graph_cg_rejects_stale_placeholder(config_context) -> None:
    """Stale placeholders are rejected."""
    with pytest.raises(ValueError):
        expand_config_path("${GRAPH_CG_RAW_DIR}/matrix.mtx", config_context)


def test_expand_unknown_neuralls_raises(config_context) -> None:
    """Unknown placeholders are rejected."""
    with pytest.raises(ValueError):
        expand_config_path("${NEURALLS_UNKNOWN}/matrix.mtx", config_context)


def test_relative_path_anchored_to_config_dir(config_context) -> None:
    """Relative paths are anchored to the config directory."""
    value = expand_config_path("relative/path.mtx", config_context)
    assert value == str((config_context.config_dir / "relative/path.mtx").resolve())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdlib expanduser prefers USERPROFILE over HOME on Windows",
)
def test_tilde_expansion(config_context, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tilde expansion follows stdlib expanduser precedence (HOME on POSIX)."""
    home = ROOT / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(ROOT / "windows-home"))
    value = expand_config_path("~/matrix.mtx", config_context)
    assert value == str((home / "matrix.mtx").resolve())


def test_glob_wildcard_preserved(
    config_context,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Glob suffixes are preserved after prefix resolution."""
    value = expand_config_glob("${NEURALLS_OUTPUT_DIR}/*.mtx", config_context)
    assert value == str(neuralls_settings.output_dir / "*.mtx")


def test_expand_raw_dir_forward_slash_separator_produces_valid_path() -> None:
    """Forward slash in template after ${NEURALLS_RAW_DIR} (Windows convention) yields an absolute path.

    On Windows the raw_dir string has backslashes; pathlib normalises the mixed separators.
    This test guards the cross-platform invariant: the result is always a non-empty absolute path
    with no double-slash prefix that would break UNC paths.
    """
    from neuralls.platform.config.context import ConfigContext

    raw = ROOT / "raw"
    settings = NeurallsSettings(
        _env_file=[],
        raw_dir=raw,
        processed_dir=ROOT / "processed",
        output_dir=ROOT / "output",
    )
    config_path = ROOT / "cfg.toml"
    ctx = ConfigContext(config_path=config_path, settings=settings)

    result = expand_config_path("${NEURALLS_RAW_DIR}/SpectralData/matrix.txt", ctx)

    assert Path(result).is_absolute()
    assert not result.startswith("//")
    assert "SpectralData" in result
    assert "matrix.txt" in result


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_expand_neuralls_raw_dir_preserves_windows_mapped_drive_prefix() -> None:
    """Placeholder expansion should keep mapped-drive roots in drive-letter form."""
    settings = NeurallsSettings(
        _env_file=[],
        raw_dir=r"M:\shared\neuralls\raw",
        processed_dir=ROOT / "processed",
        output_dir=ROOT / "output",
    )
    config_path = ROOT / "cfg.toml"
    ctx = ConfigContext(config_path=config_path, settings=settings)

    result = expand_config_path("${NEURALLS_RAW_DIR}/SpectralData/matrix.txt", ctx)

    assert result == r"M:\shared\neuralls\raw\SpectralData\matrix.txt"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_expand_neuralls_raw_dir_rejects_windows_mapped_drive_prefix_on_posix() -> None:
    """Placeholder expansion should reject mapped-drive roots on POSIX."""
    ctx = ConfigContext(
        config_path=ROOT / "cfg.toml",
        settings=SimpleNamespace(
            raw_dir=r"M:\shared\neuralls\raw",
            processed_dir=ROOT / "processed",
            output_dir=ROOT / "output",
        ),
    )

    with pytest.raises(ValueError):
        expand_config_path("${NEURALLS_RAW_DIR}/SpectralData/matrix.txt", ctx)


def test_expand_config_path_delegates_windows_unc_to_shared_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows UNC inputs should route through the shared DLKit-backed resolver."""
    captured: dict[str, object] = {}

    def _fake_resolve_local_path(value: Path | str, *, base_dir: Path) -> Path:
        captured["value"] = value
        captured["base_dir"] = base_dir
        return ROOT / "resolved" / "matrix.txt"

    ctx = ConfigContext(
        config_path=ROOT / "configs" / "dataset.toml",
        settings=SimpleNamespace(
            raw_dir=None,
            processed_dir=ROOT / "processed",
            output_dir=ROOT / "output",
        ),
    )
    monkeypatch.setattr(context_module, "resolve_local_path", _fake_resolve_local_path)

    result = expand_config_path(
        r"\\192.168.1.51\MGroup\shared\neuralls\raw\SpectralData\matrix.txt",
        ctx,
    )

    assert result == str((ROOT / "resolved" / "matrix.txt").resolve())
    assert captured["value"] == r"\\192.168.1.51\MGroup\shared\neuralls\raw\SpectralData\matrix.txt"
    assert captured["base_dir"] == ctx.config_dir
