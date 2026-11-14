"""Aggregate metric report for a completed training sweep.

Reads each assignment's MLflow metrics, renders the bar chart and label map,
and uploads both onto the sweep's session parent run.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.application.models import AssignmentResult, AssignmentSweepResult
from neuralls.platform.reporting.figure_formats import FigureTarget, figure_files
from neuralls.platform.reporting.figure_text import Legend, training_legend
from neuralls.platform.reporting.metric_bars import plot_method_bars
from neuralls.platform.tracking.mlflow_client import (
    fetch_mlflow_metrics,
    log_batch_artifacts_to_mlflow,
)
from neuralls.platform.tracking.run_names import render_training_label


def _build_label_map(results: Sequence[AssignmentResult]) -> dict[str, dict[str, str | None]]:
    """Build a mapping from assignment_id to its identity and rendered label, for the label map."""
    return {
        result.assignment_id: {
            "assignment_id": result.assignment_id,
            "assignment_label": render_training_label(result.presentation),
            "mlflow_run_id": result.mlflow_run_id,
        }
        for result in results
    }


@dataclass(frozen=True, slots=True)
class _MetricReport:
    """Everything the sweep's aggregate report renders, computed but not yet written.

    Attributes:
        labels: Assignment ids that reported the metric, in sweep order.
        values: The reported metric values, aligned with `labels`.
        legend: Bar label per plotted assignment id, rendered from its training presentation.
        missing: Rendered descriptions of assignments lacking the metric.
        label_map: Full identity of every assignment, plotted or not.
    """

    labels: tuple[str, ...]
    values: tuple[float, ...]
    legend: Legend
    missing: tuple[str, ...]
    label_map: dict[str, dict[str, str | None]]

    @property
    def has_plot(self) -> bool:
        """Whether at least one assignment reported the metric."""
        return bool(self.labels)


def _fetch_assignment_metrics(
    results: Sequence[AssignmentResult],
    tracking_uri: str,
) -> list[Mapping[str, float]]:
    """Read each assignment's MLflow metrics, positionally aligned with *results*."""
    return [
        fetch_mlflow_metrics(result.mlflow_run_id, tracking_uri)
        if result.mlflow_run_id is not None
        else {}
        for result in results
    ]


def _build_metric_report(
    results: Sequence[AssignmentResult],
    metrics_per_result: Sequence[Mapping[str, float]],
    *,
    metric: str,
) -> _MetricReport:
    """Compute the sweep's aggregate report from already-fetched metrics (pure)."""
    labels: list[str] = []
    values: list[float] = []
    missing: list[str] = []

    for result, metrics in zip(results, metrics_per_result, strict=True):
        if metric in metrics:
            labels.append(result.assignment_id)
            values.append(metrics[metric])
            continue
        missing.append(render_training_label(result.presentation))

    legend = training_legend(
        {
            result.assignment_id: result.presentation
            for result in results
            if result.assignment_id in labels
        }
    )
    return _MetricReport(
        labels=tuple(labels),
        values=tuple(values),
        legend=legend,
        missing=tuple(missing),
        label_map=_build_label_map(results),
    )


def _upload_metric_report(
    report: _MetricReport,
    *,
    metric: str,
    tracking_uri: str,
    parent_run_id: str,
) -> None:
    """Stage the report's plot and label map in a scratch dir and upload both to MLflow."""
    plot_stem = f"batch_metric_{metric.replace('/', '_')}"
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        if report.has_plot:
            plot_method_bars(
                report.legend,
                report.labels,
                report.values,
                metric_name=metric,
                target=FigureTarget(work_root, plot_stem),
            )

        (work_root / "batch_training_labels.json").write_text(
            json.dumps(report.label_map, indent=2),
            encoding="utf-8",
        )

        log_batch_artifacts_to_mlflow(
            tracking_uri=tracking_uri,
            run_id=parent_run_id,
            work_root=work_root,
            flat_files=(*figure_files(plot_stem), "batch_training_labels.json"),
        )


def write_metric_report(sweep_result: AssignmentSweepResult, *, metric: str) -> bool:
    """Build the sweep's aggregate metric plot and label map, and upload both to MLflow.

    Stages the bar chart and label map in a scratch directory and uploads them
    via `log_batch_artifacts_to_mlflow` onto the sweep's session parent run —
    nothing is left on local disk, matching every other artifact this sweep
    produces, which all live under MLflow rather than a local output directory.

    Args:
        sweep_result: Completed `run_assignment_sweep()` result.
        metric: Metric key to plot across assignments.

    Returns:
        True if a comparison plot was produced (metric present for >=1 result).
    """
    if sweep_result.tracking_uri is None or sweep_result.parent_run_id is None:
        logger.warning(
            "No MLflow session parent run for this sweep; skipping metric report upload."
        )
        return False

    report = _build_metric_report(
        sweep_result.results,
        _fetch_assignment_metrics(sweep_result.results, sweep_result.tracking_uri),
        metric=metric,
    )
    if report.missing:
        logger.warning("Metric '{}' missing for assignments: {}", metric, ", ".join(report.missing))
    _upload_metric_report(
        report,
        metric=metric,
        tracking_uri=sweep_result.tracking_uri,
        parent_run_id=sweep_result.parent_run_id,
    )
    return report.has_plot
