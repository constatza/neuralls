"""Evaluate trained assignments declared in one case config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.evaluation import eval_batch, write_eval_metric_report
from neuralls.composition.config import load_case_settings
from neuralls.composition.tracking.services import MLFLOW_TRACKING
from neuralls.shared.constants import EXIT_FAILURE


def eval_case_batch(
    config: CaseConfigArgument,
    metric: Annotated[
        str, typer.Option(help="Evaluation metric key to plot across assignments.")
    ] = "mae",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory for per-assignment local staging (downloaded checkpoints, "
                "predictions, figures). Defaults to output_dir/eval/. The batch metric "
                "plot and label map are logged to MLflow, not written here."
            )
        ),
    ] = None,
    assignment: Annotated[
        list[str] | None,
        typer.Option(
            "--assignment",
            help="Assignment id to evaluate. Repeat to evaluate multiple assignments.",
        ),
    ] = None,
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Evaluate trained assignment checkpoints on their logged test splits."""
    try:
        settings = load_case_settings(config, env_file, profile=profile)
        cfg, _ = load_validated_case_config(config, settings)
        batch = eval_batch(
            cfg=cfg,
            configs_dir=config.resolve().parent,
            settings=settings,
            output_root=output_dir,
            case_config_path=config.resolve(),
            assignment_ids=assignment,
            tracking=MLFLOW_TRACKING,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error during batch evaluation: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    plotted = write_eval_metric_report(batch, metric=metric)
    if plotted:
        typer.echo(f"Logged batch metric plot for '{metric}' to MLflow run {batch.parent_run_id}.")
    else:
        typer.echo(f"No data to plot for metric '{metric}'.", err=True)

    typer.echo(f"Logged eval label map to MLflow run {batch.parent_run_id}.")
