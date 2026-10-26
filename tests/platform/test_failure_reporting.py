"""Blocking workflow failures retain exact context in the active MLflow run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
    assert logged_context["schema"] == "comparison-workflow-failure-v1"
    assert logged_context["scope"] == "method"
    assert logged_context["phase"] == ComparisonMethodPhase.EXECUTION
    assert logged_context["method"] == context.method
    assert logged_context["config_digest"] == context.config_digest
    assert logged_context["system_ids"] == list(context.system_ids)
    assert logged_context["exception_type"] == RuntimeError.__name__
    _, traceback_path = log_text.call_args.args
    assert traceback_path == "failure/traceback.txt"
    assert set_tags.call_args.args[0] == {
        "failure.phase": "execution",
        "failure.scope": "method",
        "failure.exception_type": "RuntimeError",
        "failure.method": "amg",
    }


def test_mlflow_failure_sink_diagnostic_error_is_best_effort() -> None:
    context = ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.REPORTING,
        case_config_path=Path("case.toml"),
    )
    with (
        patch(
            "neuralls.platform.tracking.failure_reporting.mlflow.log_dict",
            side_effect=RuntimeError("tracking unavailable"),
        ),
        patch("neuralls.platform.tracking.failure_reporting.logger.warning") as warning,
    ):
        MLflowWorkflowFailureSink().record(context, _captured_error())

    warning.assert_called_once()
