"""Render the figures and metrics of a finished comparison session."""

from __future__ import annotations

import os
from pathlib import Path
from typing import assert_never

import typer
from loguru import logger

from neuralls.cli.batch import expand_case_configs, run_each
from neuralls.cli.errors import fail_workflow
from neuralls.cli.options import CaseConfigsArgument, EnvFileOption, ForceOption, ProfileOption
from neuralls.composition.assignments.comparison_report import (
    ReportCreated,
    ReportReused,
    run_comparison_report,
)
from neuralls.composition.config import CaseConfigPath, load_case_settings
from neuralls.composition.tracking.services import MLFLOW_TRACKING
from neuralls.shared.constants import EXIT_FAILURE
from neuralls.shared.types import SessionRunId

os.environ.setdefault("MPLBACKEND", "Agg")


def report_case(
    configs: CaseConfigsArgument,
    force: ForceOption = False,
    session_run_id: str | None = typer.Option(
        None, "--session-run-id", help="Report this comparison session instead of the newest."
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Draw and log the report of the newest finished comparison session of each case config."""

    def run_one(config: Path) -> None:
        try:
            settings = load_case_settings(env_file, profile=profile)
            outcome = run_comparison_report(
                CaseConfigPath.resolve(config),
                force=force,
                session_run_id=None if session_run_id is None else SessionRunId(session_run_id),
                settings=settings,
                tracking=MLFLOW_TRACKING,
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
        logger.info(
            f"{verb} report run {outcome.report_run_id} for session {outcome.session_run_id}."
        )

    paths = expand_case_configs(configs)
    if session_run_id is not None and len(paths) > 1:
        typer.echo("Error: --session-run-id names one session; give one case config.", err=True)
        raise typer.Exit(code=EXIT_FAILURE)
    run_each(paths, run_one)
