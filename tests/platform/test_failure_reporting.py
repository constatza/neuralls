"""Blocking workflow failures retain exact context in the active MLflow run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from loguru import logger

from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.shared.types import PreconditionerName
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
    ComparisonMethodFailureContext,
    ComparisonMethodPhase,
)


def _captured_error() -> RuntimeError:
    try:
        raise RuntimeError("injected workload failure")
    except RuntimeError as exc:
        return exc


def test_mlflow_failure_sink_records_exact_context_and_traceback() -> None:
    context = ComparisonMethodFailureContext(
        phase=ComparisonMethodPhase.EXECUTION,
        method=PreconditionerName("amg"),
        config_digest="sha256:" + "1" * 64,
        system_ids=("system-1", "system-2"),
    )

    with (
        patch("neuralls.platform.tracking.failure_reporting.mlflow.log_dict") as log_dict,
        patch("neuralls.platform.tracking.failure_reporting.mlflow.log_text") as log_text,
        patch("neuralls.platform.tracking.failure_reporting.mlflow.set_tags") as set_tags,
    ):
        MLflowWorkflowFailureSink().record(context, _captured_error())

    logged_context, context_path = log_dict.call_args.args
    assert context_path == "failure/context.json"
    assert logged_context == {
        "schema": "comparison-workflow-failure-v1",
        "scope": "method",
        "phase": "execution",
        "method": "amg",
        "config_digest": "sha256:" + "1" * 64,
        "system_ids": ["system-1", "system-2"],
        "exception_type": "RuntimeError",
        "message": "injected workload failure",
    }
    traceback_text, traceback_path = log_text.call_args.args
    assert traceback_path == "failure/traceback.txt"
    assert "RuntimeError: injected workload failure" in traceback_text
    assert "_captured_error" in traceback_text
    assert set_tags.call_args.args[0] == {
        "failure.phase": "execution",
        "failure.scope": "method",
        "failure.exception_type": "RuntimeError",
        "failure.method": "amg",
    }


def test_mlflow_failure_sink_diagnostic_error_is_best_effort() -> None:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    context = ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.REPORTING,
        case_config_path=Path("case.toml"),
    )
    try:
        with patch(
            "neuralls.platform.tracking.failure_reporting.mlflow.log_dict",
            side_effect=RuntimeError("tracking unavailable"),
        ):
            MLflowWorkflowFailureSink().record(context, _captured_error())
    finally:
        logger.remove(sink_id)

    assert any("tracking unavailable" in message for message in messages)
    assert any("injected workload failure" in message for message in messages)
