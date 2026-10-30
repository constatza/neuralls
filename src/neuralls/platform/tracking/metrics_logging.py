"""MLflow tracking setup and per-method metric logging for comparison workflows."""

from __future__ import annotations

from collections.abc import Mapping

import mlflow

from neuralls.domain.presentation.methods import (
    PreparedMethodPresentation,
)
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.statistics import MethodComparison, PairedComparison
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment
from neuralls.shared.types import PreconditionerName


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


def log_convergence_metrics(convergence: Mapping[PreconditionerName, ConvergenceSummary]) -> None:
    """Log explicit planned, attempted, and converged counts."""
    for name, summary in convergence.items():
        mlflow.log_metric(parent_run_metric_key("n_planned_rhs", name), summary.n_planned_rhs)
        mlflow.log_metric(parent_run_metric_key("n_attempted_rhs", name), summary.n_attempted_rhs)
        mlflow.log_metric(parent_run_metric_key("n_converged", name), summary.n_converged)
        mlflow.log_metric(parent_run_metric_key("success_rate", name), summary.success_rate)
        if summary.iterations is not None:
            _log_optional_metrics(
                name,
                {
                    "iterations_mean": summary.iterations.mean,
                    "iterations_median": summary.iterations.median,
                    "iterations_std": summary.iterations.std,
                },
            )


def _log_optional_metrics(preconditioner_name: str, metrics: Mapping[str, float | None]) -> None:
    """Log each defined (non-`None`) metric under the method-namespaced key."""
    for metric, value in metrics.items():
        if value is not None:
            mlflow.log_metric(parent_run_metric_key(metric, preconditioner_name), value)


def _paired_metrics(paired: PairedComparison) -> dict[str, float | None]:
    """Failure counts, median ratio, and matrix-cluster interval."""
    interval = paired.median_speed_ratio_ci
    return {
        "n_both_solved": float(paired.n_both_solved),
        "n_reference_only": float(paired.n_reference_only),
        "n_method_only": float(paired.n_method_only),
        "speed_ratio_median": paired.median_speed_ratio,
        "speed_ratio_ci_low": interval.low if interval is not None else None,
        "speed_ratio_ci_high": interval.high if interval is not None else None,
    }


def log_comparison_metrics(comparisons: Mapping[PreconditionerName, MethodComparison]) -> None:
    """Log each method's failure-aware reference/method speed ratios."""
    for name, comparison in comparisons.items():
        _log_optional_metrics(name, _paired_metrics(comparison.paired))


def log_configured_method_facts(prepared: PreparedMethodPresentation) -> None:
    """Log every configured fact of a method as stable parameters, before it executes."""
    mlflow.log_params(encode_method_parameters(prepared))
