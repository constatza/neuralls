"""Render the figures and metrics of a finished comparison session."""

from __future__ import annotations

import os
from typing import assert_never

import typer
from loguru import logger

from neuralls.cli.errors import fail_workflow
from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.comparison_report import (
    ReportCreated,
    ReportReused,
    run_comparison_report,
)
from neuralls.composition.config import load_case_settings
from neuralls.shared.types import SessionRunId

os.environ.setdefault("MPLBACKEND", "Agg")


def report_case(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False, "--force", "-f", help="Render again even if a matching report already exists."
    ),
    session_run_id: str | None = typer.Option(
        None, "--session-run-id", help="Report this comparison session instead of the newest."
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Draw and log the report of the newest finished comparison session of one case config."""
    try:
        settings = load_case_settings(config, env_file, profile=profile)
        outcome = run_comparison_report(
            config,
            force=force,
            session_run_id=None if session_run_id is None else SessionRunId(session_run_id),
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        fail_workflow("Report workflow", exc)
    match outcome:
        case ReportReused():
            verb = "Reused"
        case ReportCreated():
            verb = "Wrote"
        case _ as unreachable:
            assert_never(unreachable)
    logger.info(f"{verb} report run {outcome.report_run_id} for session {outcome.session_run_id}.")
