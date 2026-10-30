"""Shared option aliases for the public CLI surface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neuralls.shared.constants import EXIT_FAILURE

ProfileOption = Annotated[
    str | None,
    typer.Option(
        "--profile",
        "-p",
        help="Named config profile (overrides NEURALLS_PROFILE env var).",
    ),
]

EnvFileOption = Annotated[
    Path | None,
    typer.Option(
        "--env-file",
        help="Optional env file to load before config resolution.",
    ),
]

CaseConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--case-config",
        help="Path to case config TOML.",
    ),
]

CaseConfigArgument = Annotated[
    Path,
    typer.Argument(help="Path to a case config TOML."),
]

ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Ignore a matching existing result and run again.",
    ),
]


def require_existing_config(config: Path) -> None:
    """Exit with a clear message when the case config path does not exist."""
    if config.exists():
        return
    typer.echo(f"Error: Config file not found: {config}", err=True)
    raise typer.Exit(code=EXIT_FAILURE)
