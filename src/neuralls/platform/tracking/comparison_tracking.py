"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

from neuralls.domain.solver.cost_metrics import (
    peak_memory_per_dof,
    setup_time_per_dof,
    time_per_dof_per_iteration,
)
from neuralls.domain.solver.models.result import CGComparisonResult, ComparisonResult
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


def parent_run_metric_key(metric: str, preconditioner_name: str) -> str:
    """Build the parent-run key for one preconditioner's metric.

    MLflow's metric namespace is flat strings only — no structured/tuple
    keys — so every per-preconditioner metric on the parent run is named
    ``"<metric>/<preconditioner>"`` (MLflow's own UI renders ``"/"`` as a
    tree, which is why this convention exists at all). This is the single
    place that builds such a key: callers (including tests asserting on
    logged metric names) should use this instead of hand-reconstructing the
    separator themselves, so the convention can't silently drift between
    production code and its tests.

    Args:
        metric: Bare metric name, e.g. ``"iterations"``, ``"setup_time_s"``.
        preconditioner_name: Preconditioner's result key (sanitized here).

    Returns:
        The full parent-run metric key, e.g. ``"iterations/jacobi"``.
    """
    return f"{metric}/{sanitize_metric_key_segment(preconditioner_name)}"


def log_comparison_result_metrics(
    result: ComparisonResult,
    *,
    child_run_tags: Mapping[str, Mapping[str, str]],
) -> None:
    """Log parent-run metrics and nested child-run metrics for a comparison result."""
    system_size = result.matrix_shape[0] if result.matrix_shape is not None else None

    for name, cg in result.results.items():
        mlflow.log_metric(parent_run_metric_key("iterations", name), cg.iterations)
        mlflow.log_metric(parent_run_metric_key("final_residual", name), cg.residual)
        mlflow.log_metric(parent_run_metric_key("converged", name), int(cg.converged))
        for key, value in _cost_metrics(cg, system_size=system_size).items():
            mlflow.log_metric(parent_run_metric_key(key, name), value)

        with mlflow.start_run(run_name=name, nested=True, tags=dict(child_run_tags[name])):
            for step, residual in enumerate(cg.residual_history_rel):
                mlflow.log_metric("residual", residual, step=step)
            for step, energy_error in enumerate(cg.error_history_a_rel or []):
                mlflow.log_metric("energy_error", energy_error, step=step)
            mlflow.log_metric("iterations", cg.iterations)
            mlflow.log_metric("final_residual", cg.residual)
            mlflow.log_metric("converged", int(cg.converged))
            for key, value in _cost_metrics(cg, system_size=system_size).items():
                mlflow.log_metric(key, value)

    if result.recommendations.overall_best is not None:
        mlflow.log_param("best_preconditioner", result.recommendations.overall_best.label)


def _cost_metrics(cg: CGComparisonResult, *, system_size: int | None) -> dict[str, float]:
    """Build the cost-metric dict for one preconditioner's result, omitting unset values.

    Shared between the parent-run (namespaced) and nested child-run (bare)
    logging loops in ``log_comparison_result_metrics`` so the metric set never
    drifts between the two.

    Args:
        cg: Comparison result to derive cost metrics from.
        system_size: Linear-system size ``n``, or ``None`` if unavailable —
            size-normalized metrics are skipped in that case.

    Returns:
        Mapping of metric name to value, containing only measured metrics.
    """
    metrics: dict[str, float] = {}
    if cg.setup_time_seconds is not None:
        metrics["setup_time_s"] = cg.setup_time_seconds
    if cg.solve_time_seconds is not None:
        metrics["solve_time_s"] = cg.solve_time_seconds
    if cg.avg_iteration_time_seconds is not None:
        metrics["avg_iteration_time_s"] = cg.avg_iteration_time_seconds
    if cg.peak_memory_bytes is not None:
        metrics["peak_memory_bytes"] = float(cg.peak_memory_bytes)
    metrics.update(_normalized_cost_metrics(cg, system_size=system_size))
    return metrics


def _normalized_cost_metrics(
    cg: CGComparisonResult, *, system_size: int | None
) -> dict[str, float]:
    """Size-normalized cost metrics, empty when `system_size` is unavailable."""
    if system_size is None:
        return {}
    metrics: dict[str, float] = {}
    normalized_time = time_per_dof_per_iteration(cg, system_size=system_size)
    if normalized_time is not None:
        metrics["time_per_dof_per_iteration"] = normalized_time
    normalized_setup_time = setup_time_per_dof(cg, system_size=system_size)
    if normalized_setup_time is not None:
        metrics["setup_time_per_dof"] = normalized_setup_time
    normalized_memory = peak_memory_per_dof(cg, system_size=system_size)
    if normalized_memory is not None:
        metrics["peak_memory_per_dof"] = normalized_memory
    return metrics


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
