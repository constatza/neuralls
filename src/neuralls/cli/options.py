"""Shared option aliases for the public CLI surface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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

CaseConfigsArgument = Annotated[
    list[str],
    typer.Argument(
        help="Case config TOMLs or glob patterns (expanded here, also on Windows); run in order.",
        show_default=False,
    ),
]

ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Ignore a matching existing result and run again.",
    ),
]
