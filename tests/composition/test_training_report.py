"""Tests for write_metric_report / AssignmentSweepResult — training's aggregate
batch reporting, extracted from the retired multi_training.py to live directly
alongside run_assignment_sweep (the one training-stage implementation shared by
`neuralls train` and `neuralls run`).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from neuralls.application.models import AssignmentResult, AssignmentSweepResult
from neuralls.composition.assignments.training_batch import write_metric_report


def _sweep_result(
    *results: AssignmentResult,
    tracking_uri: str | None = "sqlite:///t.db",
    parent_run_id: str | None = "parent-1",
) -> AssignmentSweepResult:
    return AssignmentSweepResult(
        results=list(results), tracking_uri=tracking_uri, parent_run_id=parent_run_id
    )


def test_returns_false_and_skips_upload_when_no_session_parent_run() -> None:
    sweep = _sweep_result(
        AssignmentResult(
            assignment_id="a", assignment_display_name="A", status="Success", mlflow_run_id="run-a"
        ),
        tracking_uri=None,
        parent_run_id=None,
    )
    with (
        patch("neuralls.composition.assignments.training_batch.fetch_mlflow_metrics") as fetch,
        patch(
            "neuralls.composition.assignments.training_batch.log_batch_artifacts_to_mlflow"
        ) as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is False
    fetch.assert_not_called()
    upload.assert_not_called()


def test_returns_false_when_no_assignment_has_the_metric() -> None:
    sweep = _sweep_result(
        AssignmentResult(
            assignment_id="a", assignment_display_name="A", status="Success", mlflow_run_id="run-a"
        ),
    )
    with (
        patch(
            "neuralls.composition.assignments.training_batch.fetch_mlflow_metrics",
            return_value={},
        ),
        patch(
            "neuralls.composition.assignments.training_batch.log_batch_artifacts_to_mlflow"
        ) as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is False
    upload.assert_called_once()  # the label map is still uploaded even with nothing to plot


def test_plots_and_uploads_when_metric_present_for_at_least_one_assignment() -> None:
    sweep = _sweep_result(
        AssignmentResult(
            assignment_id="a", assignment_display_name="A", status="Success", mlflow_run_id="run-a"
        ),
        AssignmentResult(
            assignment_id="b", assignment_display_name="B", status="Success", mlflow_run_id="run-b"
        ),
    )

    def _fetch(run_id: str, tracking_uri: str) -> dict[str, float]:
        return {"eval/mae": 0.1} if run_id == "run-a" else {}

    with (
        patch(
            "neuralls.composition.assignments.training_batch.fetch_mlflow_metrics",
            side_effect=_fetch,
        ),
        patch("neuralls.composition.assignments.training_batch.plot_metric_comparison") as plot_fn,
        patch(
            "neuralls.composition.assignments.training_batch.log_batch_artifacts_to_mlflow"
        ) as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is True
    plot_fn.assert_called_once()
    plot_kwargs = plot_fn.call_args.kwargs
    assert plot_kwargs["labels"] == ["a"]
    assert plot_kwargs["values"] == [0.1]
    upload.assert_called_once()
    assert upload.call_args.kwargs["run_id"] == "parent-1"
    assert upload.call_args.kwargs["tracking_uri"] == "sqlite:///t.db"


def test_label_map_includes_every_result_regardless_of_metric_presence(
    tmp_path: Path,
) -> None:
    sweep = _sweep_result(
        AssignmentResult(
            assignment_id="a", assignment_display_name="A", status="Success", mlflow_run_id="run-a"
        ),
        AssignmentResult(
            assignment_id="b",
            assignment_display_name="B",
            status="Failed",
            mlflow_run_id=None,
            error="boom",
        ),
    )
    captured_label_map: dict[str, dict[str, str | None]] = {}

    def _capture_upload(*, tracking_uri: str, run_id: str, work_root: Path, flat_files) -> None:
        del tracking_uri, run_id
        assert "batch_training_labels.json" in flat_files
        captured_label_map.update(
            json.loads((work_root / "batch_training_labels.json").read_text())
        )

    with (
        patch(
            "neuralls.composition.assignments.training_batch.fetch_mlflow_metrics",
            return_value={},
        ),
        patch(
            "neuralls.composition.assignments.training_batch.log_batch_artifacts_to_mlflow",
            side_effect=_capture_upload,
        ),
    ):
        write_metric_report(sweep, metric="eval/mae")

    assert set(captured_label_map) == {"a", "b"}
    assert captured_label_map["a"]["mlflow_run_id"] == "run-a"
    assert captured_label_map["b"]["mlflow_run_id"] is None
