"""Run every comparison profile declared in one case config."""

from __future__ import annotations

import os

import typer
from loguru import logger

from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    ComparisonResult,
)
from neuralls.composition.config import load_case_settings
from neuralls.shared.constants import EXIT_FAILURE, SYMBOL_CHECKMARK

os.environ.setdefault("MPLBACKEND", "Agg")


def _log_comparison_results(result: ComparisonResult) -> None:
    logger.info(f"Available preconditioners: {result.preconditioners}")
    if result.summary:
        logger.info("=" * 60)
        logger.info(result.summary)
        logger.info("=" * 60)


def _log_outcomes(outcomes: list[ComparisonOutcome]) -> None:
    successful = [outcome for outcome in outcomes if outcome.success]
    failed = [outcome for outcome in outcomes if not outcome.success]
    partial = [outcome for outcome in successful if outcome.warnings]

    for outcome in outcomes:
        logger.info("=" * 80)
        logger.info(f"Comparison: {outcome.comparison_display_name}")
        logger.info("=" * 80)
        for warning in outcome.warnings:
            logger.warning(warning)
        if not outcome.success:
            logger.error(f"Comparison failed: {outcome.error}")
        elif outcome.payload:
            _log_comparison_results(outcome.payload)
        else:
            # Success with no payload means the reuse-check found a matching
            # comparison already run and skipped recomputing it.
            logger.info("Using existing MLflow comparison result (skipped rerun).")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total comparisons: {len(outcomes)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            logger.error(f"  x {item.comparison_id}: {item.error}")
        raise typer.Exit(code=EXIT_FAILURE)
    if partial:
        logger.warning(
            f"{len(partial)}/{len(successful)} comparisons completed with skipped "
            "preconditioners — see warnings above for what was dropped."
        )
        return
    logger.info(f"{SYMBOL_CHECKMARK} All comparisons completed successfully!")


def compare_case(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Rerun every comparison even if a matching one already exists.",
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Benchmark classical and neural preconditioners for one case config."""
    params = ComparisonParams(force=force)

    try:
        settings = load_case_settings(config, env_file, profile=profile)
        outcomes = run_comparison_batch(config, params, settings)
    except (ValueError, KeyError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE) from exc

    _log_outcomes(outcomes)
