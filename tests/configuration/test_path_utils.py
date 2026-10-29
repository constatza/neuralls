"""Tests for DLKit-backed local path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dlkit.infrastructure.io import url_resolver

from neuralls.platform.config.resolution import resolve_local_path

pytestmark = pytest.mark.deferred_io("config-loading")


def test_resolve_local_path_accepts_file_uri(tmp_path: Path) -> None:
    """file:// URIs round-trip through the shared local resolver."""
    expected = (tmp_path / "mlartifacts" / "run-1" / "artifacts").resolve()
    uri = url_resolver.build_uri(expected, scheme="file")

    resolved = resolve_local_path(uri, base_dir=Path.cwd())

    assert resolved == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_resolve_local_path_preserves_windows_mapped_drive_paths() -> None:
    """Mapped-drive paths should stay as drive-letter paths instead of resolving to UNC."""
    resolved = resolve_local_path(r"M:\shared\neuralls\raw\matrix.txt", base_dir=Path.cwd())

    assert str(resolved) == r"M:\shared\neuralls\raw\matrix.txt"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_resolve_local_path_preserves_windows_unc_paths() -> None:
    """UNC paths should remain UNC paths without config-dir anchoring."""
    resolved = resolve_local_path(
        r"\\192.168.1.51\MGroup\shared\neuralls\raw\matrix.txt",
        base_dir=Path.cwd(),
    )

    assert str(resolved) == r"\\192.168.1.51\MGroup\shared\neuralls\raw\matrix.txt"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_resolve_local_path_rejects_windows_mapped_drive_paths_on_posix() -> None:
    """Mapped-drive paths should remain invalid on POSIX hosts."""
    with pytest.raises(ValueError):
        resolve_local_path(r"M:\shared\neuralls\raw\matrix.txt", base_dir=Path.cwd())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path semantics")
def test_resolve_local_path_rejects_windows_unc_paths_on_posix() -> None:
    """UNC paths should remain invalid on POSIX hosts."""
    with pytest.raises(ValueError):
        resolve_local_path(
            r"\\192.168.1.51\MGroup\shared\neuralls\raw\matrix.txt",
            base_dir=Path.cwd(),
        )
