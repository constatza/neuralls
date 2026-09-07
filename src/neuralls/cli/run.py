"""Run the full generate-train-compare pipeline for one case config."""

from __future__ import annotations

import typer

from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.case_pipeline import run_case_pipeline
from neuralls.composition.config import load_case_settings
from neuralls.shared.constants import EXIT_FAILURE


def run_case_pipeline_command(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force retraining even if a completed MLflow run already exists.",
    ),
    force_generate: bool = typer.Option(
        False,
        "--force-generate",
        help="Regenerate every dataset even if a matching one already exists.",
    ),
    force_compare: bool = typer.Option(
        False,
        "--force-compare",
        help="Rerun every comparison even if a matching one already exists.",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help="Override max training epochs for every assignment.",
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Generate datasets, train every assignment, and run comparisons for one case config."""
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        settings = load_case_settings(config, env_file, profile=profile)
        typer.echo(f"Running case pipeline from: {config}")
        if force:
            typer.echo("Force mode enabled: existing MLflow training runs will be ignored.")

        results, comparison_outcomes = run_case_pipeline(
            case_config_path=config,
            settings=settings,
            force_train=force,
            force_generate=force_generate,
            force_compare=force_compare,
            max_epochs=max_epochs,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    typer.echo("")
    typer.echo("Summary")
    typer.echo("=" * 80)
    typer.echo(f"Completed {len(results)} assignments:")
    for result in results:
        status = "SUCCESS" if result.is_success else "FAILED"
        detail = f" ({result.error})" if result.error else ""
        label = result.assignment_display_name
        if label != result.assignment_id:
            label = f"{label} [{result.assignment_id}]"
        typer.echo(f"  {status} {label}{detail}")

    if comparison_outcomes:
        typer.echo(f"Completed {len(comparison_outcomes)} comparisons:")
        for outcome in comparison_outcomes:
            status = "SUCCESS" if outcome.success else "FAILED"
            detail = f" ({outcome.error})" if outcome.error else ""
            typer.echo(f"  {status} {outcome.comparison_display_name}{detail}")

    failures = [result for result in results if not result.is_success]
    failed_comparisons = [outcome for outcome in comparison_outcomes if not outcome.success]
    if failures or failed_comparisons:
        raise typer.Exit(code=EXIT_FAILURE)
