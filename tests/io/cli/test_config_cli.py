"""Tests for the profile-management CLI."""

from __future__ import annotations

import inspect
from pathlib import Path

from typer.core import TyperGroup as Group
from typer.main import get_command
from typer.models import OptionInfo
from typer.testing import CliRunner

import neuralls.cli.config as config_cli
import neuralls.platform.config.profile as profile_module
from neuralls.cli.config import app

runner = CliRunner()


def _configure_config_path(config_file: Path) -> None:
    profile_module.USER_CONFIG_DIR = config_file.parent
    profile_module.USER_CONFIG_FILE = config_file
    config_cli.USER_CONFIG_FILE = config_file


def test_create_declares_required_root_flags() -> None:
    parameters = inspect.signature(config_cli.create).parameters

    raw_dir = parameters["raw_dir"].default
    processed_dir = parameters["processed_dir"].default
    output_dir = parameters["output_dir"].default

    assert isinstance(raw_dir, OptionInfo)
    assert raw_dir.default is ...
    assert isinstance(processed_dir, OptionInfo)
    assert processed_dir.default is ...
    assert isinstance(output_dir, OptionInfo)
    assert output_dir.default is ...


def test_create_writes_profile_when_all_root_flags_are_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tomllib

    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    monkeypatch.delenv("NEURALLS_PROFILE", raising=False)
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    output = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "create",
            "laptop",
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(processed),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    laptop = config["profiles"]["laptop"]
    assert Path(laptop["raw_dir"]).resolve() == raw.resolve()
    assert Path(laptop["processed_dir"]).resolve() == processed.resolve()
    assert Path(laptop["output_dir"]).resolve() == output.resolve()


def test_create_fails_when_a_required_root_flag_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_config_path(tmp_path / "config.toml")
    monkeypatch.delenv("NEURALLS_PROFILE", raising=False)

    result = runner.invoke(
        app,
        [
            "create",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
        ],
    )

    assert result.exit_code == 2
    assert not (tmp_path / "config.toml").exists()


def test_init_writes_template_config(tmp_path: Path) -> None:
    _configure_config_path(tmp_path / "config.toml")

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "[default]" in (tmp_path / "config.toml").read_text(encoding="utf-8")


def test_init_force_overwrites_existing_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text('[default]\nraw_dir = "/old"\n', encoding="utf-8")

    result = runner.invoke(app, ["init", "--force"])

    assert result.exit_code == 0
    assert 'raw_dir = "/path/to/raw"' in config_file.read_text(encoding="utf-8")


def test_set_updates_existing_default_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    import tomllib

    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text(
        "\n".join(
            [
                "[default]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    new_output = tmp_path / "new-output"

    result = runner.invoke(app, ["set", "output-dir", str(new_output)])

    assert result.exit_code == 0
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert Path(config["default"]["output_dir"]).resolve() == new_output.resolve()


def test_set_updates_existing_named_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    import tomllib

    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text(
        "\n".join(
            [
                "[profiles.laptop]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    new_output = tmp_path / "new-output"

    result = runner.invoke(app, ["set", "output-dir", str(new_output), "laptop"])

    assert result.exit_code == 0
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert Path(config["profiles"]["laptop"]["output_dir"]).resolve() == new_output.resolve()


def test_set_fails_for_missing_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text(
        "\n".join(
            [
                "[default]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    original_text = config_file.read_text(encoding="utf-8")

    result = runner.invoke(app, ["set", "output-dir", str(tmp_path / "new-output"), "laptop"])

    assert result.exit_code != 0
    assert config_file.read_text(encoding="utf-8") == original_text


def test_delete_removes_existing_named_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    laptop_dirs = {k: tmp_path / "mnt" / k for k in ("raw", "processed", "output")}
    config_file.write_text(
        "\n".join(
            [
                "[default]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
                "",
                "[profiles.laptop]",
                f'raw_dir = "{laptop_dirs["raw"].as_posix()}"',
                f'processed_dir = "{laptop_dirs["processed"].as_posix()}"',
                f'output_dir = "{laptop_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["delete", "laptop"])

    assert result.exit_code == 0
    config_text = config_file.read_text(encoding="utf-8")
    assert "[default]" in config_text
    assert "[profiles.laptop]" not in config_text


def test_delete_fails_for_default_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text(
        "\n".join(
            [
                "[default]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["delete", "default"])

    assert result.exit_code != 0
    assert "[default]" in config_file.read_text(encoding="utf-8")


def test_delete_fails_for_missing_named_profile(
    tmp_path: Path,
    profile_dirs: dict[str, Path],
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_config_path(config_file)
    config_file.write_text(
        "\n".join(
            [
                "[default]",
                f'raw_dir = "{profile_dirs["raw"].as_posix()}"',
                f'processed_dir = "{profile_dirs["processed"].as_posix()}"',
                f'output_dir = "{profile_dirs["output"].as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    original_text = config_file.read_text(encoding="utf-8")

    result = runner.invoke(app, ["delete", "laptop"])

    assert result.exit_code != 0
    assert config_file.read_text(encoding="utf-8") == original_text


def test_config_subcommands_are_registered() -> None:
    command = get_command(app)
    assert isinstance(command, Group)

    assert sorted(command.commands) == [
        "create",
        "delete",
        "init",
        "list",
        "path",
        "set",
        "show",
    ]
