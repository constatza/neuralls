"""Case config arguments expand into existing paths, globs included, before anything runs."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from neuralls.cli.batch import expand_case_configs
from neuralls.shared.constants import EXIT_FAILURE


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    for name in ("b.toml", "a.toml", "notes.md"):
        (tmp_path / name).write_text("", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["*.toml"], ["a.toml", "b.toml"]),
        (["b.toml", "*.toml"], ["b.toml", "a.toml"]),
        (["a.toml", "a.toml"], ["a.toml"]),
    ],
    ids=["glob-sorted", "order-kept-duplicates-dropped", "literal-duplicate"],
)
def test_expand_case_configs_expands_in_order(
    case_dir: Path, args: list[str], expected: list[str]
) -> None:
    paths = expand_case_configs([str(case_dir / arg) for arg in args])

    assert paths == [case_dir / name for name in expected]


@pytest.mark.parametrize("arg", ["*.json", "missing.toml"], ids=["unmatched-glob", "missing-file"])
def test_expand_case_configs_rejects_unresolvable_argument(case_dir: Path, arg: str) -> None:
    with pytest.raises(typer.Exit) as raised:
        expand_case_configs([str(case_dir / "a.toml"), str(case_dir / arg)])

    assert raised.value.exit_code == EXIT_FAILURE
