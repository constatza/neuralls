"""Consistent rendering for exceptions that abort a CLI workflow."""

from __future__ import annotations

import traceback
from typing import NoReturn

import typer
from loguru import logger

from neuralls.shared.constants import EXIT_FAILURE


def fail_workflow(operation: str, exc: Exception) -> NoReturn:
    """Render an uncaught workflow exception without discarding its type or notes."""
    details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    logger.opt(exception=exc).critical("{} failed", operation)
    typer.echo(f"{operation} failed [{type(exc).__name__}]: {exc}", err=True)
    notes = getattr(exc, "__notes__", ())
    if notes:
        typer.echo("\n".join(str(note) for note in notes), err=True)
    elif details != f"{type(exc).__name__}: {exc}":
        typer.echo(details, err=True)
    raise typer.Exit(code=EXIT_FAILURE) from exc
