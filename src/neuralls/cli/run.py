"""Run the full generate-train-compare pipeline for one case config."""

from __future__ import annotations

import typer

from neuralls.cli.errors import fail_workflow
from neuralls.cli.options import (
    CaseConfigArgument,
    EnvFileOption,
    ProfileOption,
    require_existing_config,
)
from neuralls.composition.assignments.case_pipeline import ForcePolicy, run_case_pipeline
from neuralls.composition.comparison.models import ComparisonEntryFailed
from neuralls.composition.config import load_case_settings
from neuralls.composition.display import assignment_label, comparison_label
from neuralls.composition.tracking.services import MLFLOW_TRACKING
from neuralls.shared.constants import EXIT_FAILURE


def run_case_pipeline_command(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force every stage — generate, train, and compare — even if a "
        "matching result already exists.",
    ),
    force_generate: bool = typer.Option(
        False,
        "--force-generate",
        help="Regenerate every dataset even if a matching one already exists.",
    ),
    force_train: bool = typer.Option(
        False,
        "--force-train",
        help="Retrain every assignment even if a completed MLflow run already exists.",
    ),
    force_compare: bool = typer.Option(
        False,
        "--force-compare",
        help="Rerun every comparison even if a matching one already exists.",
    ),
    force_expand: bool = typer.Option(
        False,
        "--force-expand",
        help="Recreate sweep-generated dataset configs even if they already exist.",
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Generate datasets, train every assignment, and run comparisons for one case config."""
    require_existing_config(config)

    try:
        settings = load_case_settings(config, env_file, profile=profile)
        typer.echo(f"Running case pipeline from: {config}")
        policy = ForcePolicy.resolve(
            everything=force,
            generate=force_generate,
            train=force_train,
            compare=force_compare,
            expand=force_expand,
        )
        if force:
            typer.echo("Force mode enabled: generation, training, and comparison will all rerun.")

        results, comparison_outcomes = run_case_pipeline(
            case_config_path=config,
            settings=settings,
            force=policy,
            tracking=MLFLOW_TRACKING,
        )
    except Exception as exc:  # noqa: BLE001
        fail_workflow("Case pipeline", exc)

    typer.echo("")
    typer.echo("Summary")
    typer.echo("=" * 80)
    typer.echo(f"Completed {len(results)} assignments:")
    for result in results:
        status = "SUCCESS" if result.is_success else "FAILED"
        detail = f" ({result.error})" if result.error else ""
        typer.echo(f"  {status} {assignment_label(result)}{detail}")

    if comparison_outcomes:
        typer.echo(f"Completed {len(comparison_outcomes)} comparisons:")
        for outcome in comparison_outcomes:
            status = "FAILED" if isinstance(outcome, ComparisonEntryFailed) else "SUCCESS"
            detail = f" ({outcome.reason})" if isinstance(outcome, ComparisonEntryFailed) else ""
            typer.echo(f"  {status} {comparison_label(outcome.run)}{detail}")

    failures = [result for result in results if not result.is_success]
    failed_comparisons = [
        outcome for outcome in comparison_outcomes if isinstance(outcome, ComparisonEntryFailed)
    ]
    if failures or failed_comparisons:
        raise typer.Exit(code=EXIT_FAILURE)
