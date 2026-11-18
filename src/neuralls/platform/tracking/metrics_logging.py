"""MLflow experiment setup and the per-method metrics of comparison report runs."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment
from neuralls.shared.types import PreconditionerName


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None,
    experiment_name: str,
) -> str:
    """Create or reuse the experiment comparison runs are written to and return its id."""
    return ensure_experiment(
        experiment_name,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_location),
    )


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


def convergence_metrics(
    convergence: Mapping[PreconditionerName, ConvergenceSummary],
) -> dict[str, float]:
    """Explicit planned, attempted, and converged counts and the iteration statistics.

    ``iterations_n_defined`` counts the systems the iteration statistics are taken over and
    ``iterations_n_planned`` the systems planned, so a statistic over few systems is visible.
    """
    metrics: dict[str, float] = {}
    for name, summary in convergence.items():
        counts: dict[str, float | None] = {
            "n_planned_rhs": summary.n_planned_rhs,
            "n_attempted_rhs": summary.n_attempted_rhs,
            "n_converged": summary.n_converged,
            "success_rate": summary.success_rate,
            count_key(MetricId.ITERATIONS, "n_defined"): summary.n_converged,
            count_key(MetricId.ITERATIONS, "n_planned"): summary.n_planned_rhs,
        }
        metrics |= _namespaced(name, counts)
        if summary.iterations is not None:
            metrics |= _namespaced(
                name,
                {
                    metric_key(MetricId.ITERATIONS, "mean"): summary.iterations.mean,
                    metric_key(MetricId.ITERATIONS, "median"): summary.iterations.median,
                    metric_key(MetricId.ITERATIONS, "std"): summary.iterations.std,
                },
            )
    return metrics


def _namespaced(preconditioner_name: str, metrics: Mapping[str, float | None]) -> dict[str, float]:
    """Each defined (non-`None`) metric under its method-namespaced key."""
    return {
        parent_run_metric_key(metric, preconditioner_name): float(value)
        for metric, value in metrics.items()
        if value is not None
    }


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


def comparison_metrics(speed_ratios: pl.DataFrame) -> dict[str, float]:
    """Each method's failure-aware reference/method speed ratios.

    ``speed_ratios`` holds ``SPEED_RATIO_SCHEMA`` rows, one per non-reference method.
    """
    metrics: dict[str, float] = {}
    for row in speed_ratios.iter_rows(named=True):
        metrics |= _namespaced(
            row["method"],
            {metric: row[column] for metric, column in _SPEED_RATIO_METRIC_COLUMNS.items()},
        )
    return metrics
