"""Several case configs run in order, and one failed case does not stop the others."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from neuralls.cli.batch import run_each
from neuralls.shared.constants import EXIT_FAILURE

CASES = [Path("cases/a.toml"), Path("cases/b.toml"), Path("cases/c.toml")]


@pytest.mark.parametrize(
    ("failing", "expected_exit"),
    [(frozenset(), None), (frozenset({CASES[0]}), EXIT_FAILURE)],
    ids=["all-succeed", "first-fails"],
)
def test_run_each_runs_every_case_and_reports_failure(
    failing: frozenset[Path], expected_exit: int | None
) -> None:
    ran: list[Path] = []

    def run_one(path: Path) -> None:
        ran.append(path)
        if path in failing:
            raise typer.Exit(code=1)

    try:
        run_each(CASES, run_one)
        exit_code = None
    except typer.Exit as exit_:
        exit_code = exit_.exit_code

    assert ran == CASES
    assert exit_code == expected_exit
