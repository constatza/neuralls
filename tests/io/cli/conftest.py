"""Shared fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def profile_dirs(tmp_path: Path) -> dict[str, Path]:
    """Raw/processed/output paths derived from tmp_path for profile config tests."""
    return {
        "raw": tmp_path / "raw",
        "processed": tmp_path / "processed",
        "output": tmp_path / "output",
    }
