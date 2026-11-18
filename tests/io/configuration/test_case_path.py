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


@pytest.mark.parametrize(
    ("relative", "experiment_name"),
    [
        (Path("configs/cases/spectral-3d-field/default.toml"), "spectral-3d-field/default"),
        (Path("rhc/pod-sweep.toml"), "rhc/pod-sweep"),
    ],
)
def test_key_is_the_case_directory_and_the_config_stem(
    relative: Path, experiment_name: str
) -> None:
    assert CaseConfigPath.resolve(relative).key.experiment_name == experiment_name
