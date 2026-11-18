"""Run one case command over several case configs, one after another.

A list of cases is what to execute now, not part of any experiment's definition, so it is
given on the command line (paths or glob patterns) instead of in a config file that lists
other configs. Patterns are expanded here because cmd and PowerShell pass `*` through
unexpanded; paths a POSIX shell already expanded arrive as plain paths.
"""

from __future__ import annotations

import glob
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import typer

from neuralls.shared.constants import EXIT_FAILURE

_GLOB_CHARACTERS = frozenset("*?[")


@dataclass(frozen=True, slots=True)
class CaseSucceeded:
    """One case config whose command finished without a failure exit."""

    path: Path


@dataclass(frozen=True, slots=True)
class CaseFailed:
    """One case config whose command exited with a non-zero code."""

    path: Path
    exit_code: int


CaseOutcome = CaseSucceeded | CaseFailed


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=EXIT_FAILURE)


def expand_case_configs(args: list[str]) -> list[Path]:
    """Expand case config arguments into existing paths, in order and without duplicates.

    Every argument is checked before any case runs, so a typo in the last config fails
    the command at once instead of after the earlier cases have run.

    Raises:
        typer.Exit: A pattern matches no file or a literal path does not exist.
    """
    paths: list[Path] = []
    for arg in args:
        if _GLOB_CHARACTERS.isdisjoint(arg):
            path = Path(arg)
            if not path.exists():
                raise _fail(f"Config file not found: {arg}")
            matches = [path]
        else:
            matches = [Path(match) for match in sorted(glob.glob(arg))]
            if not matches:
                raise _fail(f"No config file matches: {arg}")
        for match in matches:
            if match not in paths:
                paths.append(match)
    return paths


def _run_one_case(path: Path, run_one: Callable[[Path], None]) -> CaseOutcome:
    try:
        run_one(path)
    except typer.Exit as exit_:
        if exit_.exit_code != 0:
            return CaseFailed(path=path, exit_code=exit_.exit_code)
    return CaseSucceeded(path=path)


def _summary_line(outcome: CaseOutcome) -> str:
    match outcome:
        case CaseSucceeded(path=path):
            return f"  SUCCESS {path}"
        case CaseFailed(path=path, exit_code=code):
            return f"  FAILED  {path} (exit code {code})"
        case _ as unreachable:
            assert_never(unreachable)


def run_each(paths: list[Path], run_one: Callable[[Path], None]) -> None:
    """Run `run_one` on every case config in order, continuing past a failed case.

    A single config runs exactly as before. With several, each gets a header, a failed
    case does not stop the rest, and one summary closes the command, which then exits
    with a failure code if any case failed.
    """
    if len(paths) == 1:
        run_one(paths[0])
        return

    outcomes: list[CaseOutcome] = []
    for index, path in enumerate(paths, start=1):
        typer.echo(f"\n[{index}/{len(paths)}] {path}")
        outcomes.append(_run_one_case(path, run_one))

    typer.echo(f"\nCase configs: {len(outcomes)}")
    for outcome in outcomes:
        typer.echo(_summary_line(outcome))
    if any(isinstance(outcome, CaseFailed) for outcome in outcomes):
        raise typer.Exit(code=EXIT_FAILURE)
