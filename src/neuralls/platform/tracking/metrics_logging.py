"""MLflow tracking setup and per-method metric logging for comparison workflows."""

from __future__ import annotations

from collections.abc import Mapping

import mlflow
import polars as pl

from neuralls.domain.presentation.methods import (
    PreparedMethodPresentation,
)
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
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


def metric_key(metric: MetricId, field: str) -> str:
    """The bare metric name of ``field`` of ``metric``, generated from the metric catalog.

    A statistic is named after the outcomes its metric is defined on, so a key states which
    systems entered it: ``iterations_converged_mean`` is a mean over converged systems only,
    because the catalog defines iterations on no other outcome.
    """
    outcomes = "_".join(sorted(outcome.value for outcome in METRICS[metric].defined_on))
    return f"{metric}_{outcomes}_{field}"


def count_key(metric: MetricId, field: str) -> str:
    """The bare name of a count (``n_defined``, ``n_planned``) of ``metric``'s systems."""
    return f"{metric}_{field}"


def log_convergence_metrics(convergence: Mapping[PreconditionerName, ConvergenceSummary]) -> None:
    """Log explicit planned, attempted, and converged counts and the iteration statistics.

    ``iterations_n_defined`` counts the systems the iteration statistics are taken over and
    ``iterations_n_planned`` the systems planned, so a statistic over few systems is visible.
    """
    for name, summary in convergence.items():
        mlflow.log_metric(parent_run_metric_key("n_planned_rhs", name), summary.n_planned_rhs)
        mlflow.log_metric(parent_run_metric_key("n_attempted_rhs", name), summary.n_attempted_rhs)
        mlflow.log_metric(parent_run_metric_key("n_converged", name), summary.n_converged)
        mlflow.log_metric(parent_run_metric_key("success_rate", name), summary.success_rate)
        _log_optional_metrics(
            name,
            {
                count_key(MetricId.ITERATIONS, "n_defined"): summary.n_converged,
                count_key(MetricId.ITERATIONS, "n_planned"): summary.n_planned_rhs,
            },
        )
        if summary.iterations is not None:
            _log_optional_metrics(
                name,
                {
                    metric_key(MetricId.ITERATIONS, "mean"): summary.iterations.mean,
                    metric_key(MetricId.ITERATIONS, "median"): summary.iterations.median,
                    metric_key(MetricId.ITERATIONS, "std"): summary.iterations.std,
                },
            )


def _log_optional_metrics(preconditioner_name: str, metrics: Mapping[str, float | None]) -> None:
    """Log each defined (non-`None`) metric under the method-namespaced key."""
    for metric, value in metrics.items():
        if value is not None:
            mlflow.log_metric(parent_run_metric_key(metric, preconditioner_name), value)


_SPEED_RATIO_METRIC_COLUMNS = {
    "n_both_solved": "n_systems_both_solved",
    "n_reference_only": "n_reference_only",
    "n_method_only": "n_method_only",
    "n_ratio": "n_ratio",
    "n_matrices": "n_matrices",
    "speed_ratio_median": "median_ratio",
    "speed_ratio_ci_low": "ci_low",
    "speed_ratio_ci_high": "ci_high",
}
"""MLflow metric name -> ``SPEED_RATIO_SCHEMA`` column it is logged from."""


def log_comparison_metrics(speed_ratios: pl.DataFrame) -> None:
    """Log each method's failure-aware reference/method speed ratios.

    ``speed_ratios`` holds ``SPEED_RATIO_SCHEMA`` rows, one per non-reference method.
    """
    for row in speed_ratios.iter_rows(named=True):
        _log_optional_metrics(
            row["method"],
            {metric: row[column] for metric, column in _SPEED_RATIO_METRIC_COLUMNS.items()},
        )


def log_configured_method_facts(prepared: PreparedMethodPresentation) -> None:
    """Log every configured fact of a method as stable parameters, before it executes."""
    mlflow.log_params(encode_method_parameters(prepared))
