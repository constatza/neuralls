"""Aggregate metric report for a completed eval-only batch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from loguru import logger

from neuralls.composition.assignments.evaluation import EvaluationBatchResult
from neuralls.platform.reporting.figure_formats import FigureTarget, figure_files
from neuralls.platform.reporting.figure_text import training_legend
from neuralls.platform.reporting.metric_bars import plot_method_bars
from neuralls.platform.tracking.mlflow_client import log_batch_artifacts_to_mlflow
from neuralls.platform.tracking.run_names import render_training_label


def _metric_value(metrics: dict[str, float], metric: str) -> float | None:
    if metric in metrics:
        return metrics[metric]
    prefixed = f"eval/{metric}"
    return metrics.get(prefixed)


def write_eval_metric_report(
    batch: EvaluationBatchResult,
    *,
    metric: str,
) -> bool:
    """Build eval-batch summary artifacts and log them to the batch's MLflow parent run.

    Stages the comparison barplot and label map in a scratch directory and
    uploads them via ``log_batch_artifacts_to_mlflow`` — nothing is left
    on local disk, matching the batch's other artifacts, which all live under
    the session parent run rather than a non-MLflow output directory.

    Args:
        batch: Completed eval batch result.
        metric: Metric key to plot across assignments.

    Returns:
        True if a comparison plot was produced (metric present for >=1 result).
    """
    plotted_ids: list[str] = []
    values: list[float] = []
    for result in batch.results:
        value = _metric_value(result.metrics, metric)
        if value is None:
            logger.warning(
                "Metric '{}' missing for {}.", metric, render_training_label(result.presentation)
            )
            continue
        plotted_ids.append(result.assignment_id)
        values.append(value)

    plot_stem = f"batch_metric_{metric.replace('/', '_')}"
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        plotted = bool(plotted_ids)
        if plotted:
            plot_method_bars(
                training_legend(
                    {
                        result.assignment_id: result.presentation
                        for result in batch.results
                        if result.assignment_id in plotted_ids
                    }
                ),
                plotted_ids,
                values,
                metric_name=metric,
                target=FigureTarget(work_root, plot_stem),
            )

        (work_root / "batch_eval_labels.json").write_text(
            json.dumps(batch.label_map, indent=2),
            encoding="utf-8",
        )

        log_batch_artifacts_to_mlflow(
            tracking_uri=batch.tracking_uri,
            run_id=batch.parent_run_id,
            work_root=work_root,
            flat_files=(*figure_files(plot_stem), "batch_eval_labels.json"),
        )

    return plotted
