"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking for comparison runs."""
    resolved_experiment_name = experiment_name or ExperimentNamesConfig().comparison
    ensure_experiment(
        resolved_experiment_name,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_location),
    )
    mlflow.set_experiment(resolved_experiment_name)


def log_comparison_artifact_uri() -> None:
    """Log the current comparison run artifact URI."""
    mlflow.log_param("artifact_uri", mlflow.get_artifact_uri())


def log_skipped_preconditioners(warnings: Sequence[str]) -> None:
    """Log skipped-preconditioner count and reasons so drops are visible on the run.

    Each warning already carries the preconditioner name and failure reason
    (see ``resolve_preconditioner_models_with_warnings``). The count alone is not
    enough to notice a silently thinner comparison, so the full messages are
    logged as a text artifact next to the comparison plot.
    """
    if warnings:
        mlflow.log_param("skipped_preconditioners", str(len(warnings)))
        mlflow.log_text("\n".join(warnings), "skipped_preconditioners.txt")


def log_comparison_result_metrics(
    result: ComparisonResult,
    *,
    child_run_tags: Mapping[str, Mapping[str, str]],
) -> None:
    """Log parent-run metrics and nested child-run metrics for a comparison result."""
    for name, cg in result.results.items():
        safe = sanitize_metric_key_segment(name)
        mlflow.log_metric(f"iterations/{safe}", cg.iterations)
        mlflow.log_metric(f"final_residual/{safe}", cg.residual)
        mlflow.log_metric(f"converged/{safe}", int(cg.converged))

        with mlflow.start_run(run_name=name, nested=True, tags=dict(child_run_tags[name])):
            for step, residual in enumerate(cg.residual_history_rel):
                mlflow.log_metric("residual", residual, step=step)
            mlflow.log_metric("iterations", cg.iterations)
            mlflow.log_metric("final_residual", cg.residual)
            mlflow.log_metric("converged", int(cg.converged))

    if result.recommendations.overall_best is not None:
        mlflow.log_param("best_preconditioner", result.recommendations.overall_best.label)


def log_linear_system_params(result: ComparisonResult) -> None:
    """Log matrix/rhs shape and RHS provenance.

    A matrix may be paired with several different RHS sources (gaussian,
    sparse, raw, dataset, ...) across sibling ``[[comparisons]]`` entries,
    each its own MLflow run — ``rhs_source_kind`` disambiguates which one
    this particular run used.
    """
    if result.matrix_shape is not None:
        mlflow.log_param("matrix_rows", result.matrix_shape[0])
        mlflow.log_param("matrix_cols", result.matrix_shape[1])
    if result.rhs_shape is not None:
        mlflow.log_param("rhs_dim", result.rhs_shape[0])
    if result.rhs_source_kind is not None:
        mlflow.log_param("rhs_source_kind", str(result.rhs_source_kind))


def log_comparison_run_params(
    *,
    comp_run_id: str,
    comparison_id: str,
    comparison_display_name: str,
    comparison_config: str,
) -> None:
    """Log final metadata params to the active comparison MLflow run."""
    mlflow.log_param("comparison_config", comparison_config)
    mlflow.log_param("comparison_id", comparison_id)
    mlflow.log_param("comparison_display_name", comparison_display_name)
    mlflow.log_param("comp_run_id", comp_run_id)


def log_comparison_input_artifacts(artifact_dir: Path) -> None:
    """Upload the resolved comparison input system artifacts to MLflow."""
    mlflow.log_artifacts(str(artifact_dir), artifact_path="comparison_inputs")
