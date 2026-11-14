"""The load edge fills the context: symlinks below the config dir stay, `~` is the real home."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.loaders import (
    load_case_config,
    load_comparison_config,
    load_data_config,
)
from neuralls.platform.config.settings import NeurallsSettings

DATA_CONFIG = """\
id = "anchored"

[source]
matrix_path = "{matrix_path}"

[generation]

[output]
"""


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "configs"
    directory.mkdir()
    return directory


@pytest.fixture
def linked_dir(tmp_path: Path, configs_dir: Path) -> Path:
    """`configs/linked`, a symlink to a sibling of `configs`."""
    target = tmp_path / "real-target"
    target.mkdir()
    link = configs_dir / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are not available: {error}")
    return link


@pytest.fixture
def home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home directory both POSIX (`HOME`) and Windows (`USERPROFILE`) lookups find."""
    home = tmp_path / "fake-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def test_data_config_keeps_a_symlinked_directory(
    configs_dir: Path, linked_dir: Path, neuralls_settings: NeurallsSettings
) -> None:
    config_path = configs_dir / "data.toml"
    config_path.write_text(DATA_CONFIG.format(matrix_path="linked/matrix.mtx"), encoding="utf-8")

    config = load_data_config(config_path, neuralls_settings)

    assert config.source.matrix_path == str(linked_dir / "matrix.mtx")


def test_case_config_keeps_a_symlinked_directory(
    configs_dir: Path, linked_dir: Path, neuralls_settings: NeurallsSettings
) -> None:
    (linked_dir / "dataset.toml").write_text('id = "linked-dataset"\n', encoding="utf-8")
    config_path = configs_dir / "case.toml"
    config_path.write_text('[[datasets]]\npath = "linked/dataset.toml"\n', encoding="utf-8")

    case = load_case_config(config_path, neuralls_settings)

    assert [entry.path for entry in case.datasets] == [linked_dir / "dataset.toml"]


def test_data_config_expands_tilde_from_the_real_home(
    configs_dir: Path, home_dir: Path, neuralls_settings: NeurallsSettings
) -> None:
    assert Path.home() == home_dir
    config_path = configs_dir / "data.toml"
    config_path.write_text(DATA_CONFIG.format(matrix_path="~/matrix.mtx"), encoding="utf-8")

    config = load_data_config(config_path, neuralls_settings)

    assert config.source.matrix_path == str(home_dir / "matrix.mtx")


def test_case_config_expands_tilde_from_the_real_home(
    configs_dir: Path, home_dir: Path, neuralls_settings: NeurallsSettings
) -> None:
    (home_dir / "dataset.toml").write_text('id = "home-dataset"\n', encoding="utf-8")
    config_path = configs_dir / "case.toml"
    config_path.write_text('[[datasets]]\npath = "~/dataset.toml"\n', encoding="utf-8")

    case = load_case_config(config_path, neuralls_settings)

    assert [entry.path for entry in case.datasets] == [home_dir / "dataset.toml"]


COMPARISON_CONFIG = """\
[general]

[general.params]
rtol = 1e-6

[general.data]
matrix_path = "~/matrix.npy"
rhs_path = "~/rhs.npy"

[[preconditioners]]
name = "jacobi"
type = "jacobi"
"""


def test_comparison_config_expands_tilde_from_the_real_home(
    configs_dir: Path, home_dir: Path, neuralls_settings: NeurallsSettings
) -> None:
    config_path = configs_dir / "comparison.toml"
    config_path.write_text(COMPARISON_CONFIG, encoding="utf-8")

    config = load_comparison_config(config_path, neuralls_settings)

    assert config.general.data.matrix_path == home_dir / "matrix.npy"
