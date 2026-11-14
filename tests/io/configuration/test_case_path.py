"""CaseConfigPath carries the absolute spelling that session tags and report lookups share."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.case_path import CaseConfigPath


def test_rejects_a_relative_path() -> None:
    relative = Path("case.toml")

    with pytest.raises(ValueError, match="case.toml"):
        CaseConfigPath(relative)


def test_resolve_makes_a_relative_path_absolute_against_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    case = CaseConfigPath.resolve(Path("case.toml"))

    assert case.path == tmp_path.resolve() / "case.toml"
    assert case.config_dir == tmp_path.resolve()
