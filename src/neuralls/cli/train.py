"""Train every assignment declared in one case config."""

from __future__ import annotations

from typing import Annotated

import typer

from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.training_batch import (
    run_assignment_sweep,
    write_metric_report,
)
from neuralls.composition.config import load_case_settings
from neuralls.shared.constants import EXIT_FAILURE


def train_case_batch(
    config: CaseConfigArgument,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Force retraining even if a completed MLflow run already exists."
        ),
    ] = False,
    metric: Annotated[
        str, typer.Option(help="MLflow metric key to plot across assignments.")
    ] = "eval/mae",
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Train the registry-defined assignment batch and emit aggregate reporting."""
    try:
        settings = load_case_settings(config, env_file, profile=profile)
        sweep_result = run_assignment_sweep(
            case_config_path=config.resolve(),
            settings=settings,
            force=force,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error during batch training: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    plotted = write_metric_report(sweep_result, metric=metric)
    if plotted:
        typer.echo(
            f"Logged batch metric plot for '{metric}' to MLflow run {sweep_result.parent_run_id}."
        )
    else:
        typer.echo(f"No data to plot for metric '{metric}'.", err=True)

    typer.echo(f"Logged batch label map to MLflow run {sweep_result.parent_run_id}.")
