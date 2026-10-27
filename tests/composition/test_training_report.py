"""Tests for write_metric_report / AssignmentSweepResult — training's aggregate
batch reporting, extracted from the retired multi_training.py to live directly
alongside run_assignment_sweep (the one training-stage implementation shared by
`neuralls train` and `neuralls run`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralls.application.models import AssignmentResult, AssignmentSweepResult
from neuralls.composition.assignments.training_batch import write_metric_report
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.tracking.run_names import render_training_label

TRAINING = "neuralls.composition.assignments.training_batch"


def _sweep_result(
    *results: AssignmentResult,
    tracking_uri: str | None = "sqlite:///t.db",
    parent_run_id: str | None = "parent-1",
) -> AssignmentSweepResult:
    return AssignmentSweepResult(
        results=list(results), tracking_uri=tracking_uri, parent_run_id=parent_run_id
    )


@pytest.fixture
def result_of(
    training_presentation_of: Callable[[str], TrainingPresentation],
) -> Callable[..., AssignmentResult]:
    """Build an assignment result whose model is named ``method_name``."""

    def build(
        assignment_id: str, method_name: str, *, run: str | None = None, **kwargs: object
    ) -> AssignmentResult:
        return AssignmentResult(
            assignment_id=assignment_id,
            presentation=training_presentation_of(method_name),
            status="Success",
            mlflow_run_id=run,
            **kwargs,
        )

    return build


def test_returns_false_and_skips_upload_when_no_session_parent_run(
    result_of: Callable[..., AssignmentResult],
) -> None:
    sweep = _sweep_result(
        result_of("a", "Alpha", run="run-a"), tracking_uri=None, parent_run_id=None
    )
    with (
        patch(f"{TRAINING}.fetch_mlflow_metrics") as fetch,
        patch(f"{TRAINING}.log_batch_artifacts_to_mlflow") as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is False
    fetch.assert_not_called()
    upload.assert_not_called()


def test_returns_false_when_no_assignment_has_the_metric(
    result_of: Callable[..., AssignmentResult],
) -> None:
    sweep = _sweep_result(result_of("a", "Alpha", run="run-a"))
    with (
        patch(f"{TRAINING}.fetch_mlflow_metrics", return_value={}),
        patch(f"{TRAINING}.log_batch_artifacts_to_mlflow") as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is False
    upload.assert_called_once()  # the label map is still uploaded even with nothing to plot


def test_plots_and_uploads_when_metric_present_for_at_least_one_assignment(
    result_of: Callable[..., AssignmentResult],
) -> None:
    sweep = _sweep_result(result_of("a", "Alpha", run="run-a"), result_of("b", "Beta", run="run-b"))

    def _fetch(run_id: str, tracking_uri: str) -> dict[str, float]:
        return {"eval/mae": 0.1} if run_id == "run-a" else {}

    with (
        patch(f"{TRAINING}.fetch_mlflow_metrics", side_effect=_fetch),
        patch(f"{TRAINING}.plot_method_bars") as plot_fn,
        patch(f"{TRAINING}.log_batch_artifacts_to_mlflow") as upload,
    ):
        plotted = write_metric_report(sweep, metric="eval/mae")

    assert plotted is True
    plot_fn.assert_called_once()
    assert plot_fn.call_args.args[1] == ("a",)
    assert plot_fn.call_args.args[2] == (0.1,)
    upload.assert_called_once()
    assert upload.call_args.kwargs["run_id"] == "parent-1"
    assert upload.call_args.kwargs["tracking_uri"] == "sqlite:///t.db"


def test_bars_are_labelled_with_the_rendered_training_presentation_not_the_assignment_id(
    result_of: Callable[..., AssignmentResult],
) -> None:
    alpha = result_of("a-17", "Alpha", run="run-a")
    sweep = _sweep_result(alpha)

    with (
        patch(f"{TRAINING}.fetch_mlflow_metrics", return_value={"eval/mae": 0.1}),
        patch(f"{TRAINING}.plot_method_bars") as plot_fn,
        patch(f"{TRAINING}.log_batch_artifacts_to_mlflow"),
    ):
        write_metric_report(sweep, metric="eval/mae")

    legend = plot_fn.call_args.args[0]
    assert legend.label("a-17") == render_training_label(alpha.presentation)
    assert "a-17" not in legend.label("a-17")


def test_missing_metric_warning_names_the_assignment_by_its_presentation(
    result_of: Callable[..., AssignmentResult],
) -> None:
    missing = result_of("a-17", "Alpha", run="run-a")
    messages: list[str] = []
    from loguru import logger

    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        with (
            patch(f"{TRAINING}.fetch_mlflow_metrics", return_value={}),
            patch(f"{TRAINING}.log_batch_artifacts_to_mlflow"),
        ):
            write_metric_report(_sweep_result(missing), metric="eval/mae")
    finally:
        logger.remove(sink_id)

    assert any(render_training_label(missing.presentation) in text for text in messages)
    assert not any("a-17" in text for text in messages)


def test_label_map_includes_every_result_regardless_of_metric_presence(
    result_of: Callable[..., AssignmentResult],
) -> None:
    sweep = _sweep_result(
        result_of("a", "Alpha", run="run-a"),
        result_of("b", "Beta", run=None, error="boom"),
    )
    captured_label_map: dict[str, dict[str, str | None]] = {}

    def _capture_upload(*, tracking_uri: str, run_id: str, work_root: Path, flat_files) -> None:
        del tracking_uri, run_id
        assert "batch_training_labels.json" in flat_files
        captured_label_map.update(
            json.loads((work_root / "batch_training_labels.json").read_text())
        )

    with (
        patch(f"{TRAINING}.fetch_mlflow_metrics", return_value={}),
        patch(f"{TRAINING}.log_batch_artifacts_to_mlflow", side_effect=_capture_upload),
    ):
        write_metric_report(sweep, metric="eval/mae")

    assert set(captured_label_map) == {"a", "b"}
    assert captured_label_map["a"]["mlflow_run_id"] == "run-a"
    assert captured_label_map["b"]["mlflow_run_id"] is None
    assert captured_label_map["a"]["assignment_label"] == render_training_label(
        sweep.results[0].presentation
    )
