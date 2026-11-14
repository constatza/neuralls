"""Blocking workflow failures retain exact context in the active MLflow run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.shared.types import PreconditionerName
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
    ComparisonMethodFailureContext,
    ComparisonMethodPhase,
)

_SINK = "neuralls.platform.tracking.failure_reporting"


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
        patch(f"{_SINK}.mlflow.active_run", return_value=MagicMock()),
        patch("neuralls.platform.tracking.failure_reporting.mlflow.log_dict") as log_dict,
        patch("neuralls.platform.tracking.failure_reporting.mlflow.log_text") as log_text,
        patch("neuralls.platform.tracking.failure_reporting.mlflow.set_tags") as set_tags,
    ):
        MLflowWorkflowFailureSink().record(context, _captured_error())

    assert log_dict.call_args.args == (
        {
            "phase": "execution",
            "exception_type": "RuntimeError",
            "message": "injected workload failure",
            "scope": "method",
            "method": "amg",
            "config_digest": context.config_digest,
            "system_ids": ["system-1", "system-2"],
        },
        "failure/context.json",
    )
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
        patch(f"{_SINK}.mlflow.active_run", return_value=MagicMock()),
        patch(
            "neuralls.platform.tracking.failure_reporting.mlflow.log_dict",
            side_effect=RuntimeError("tracking unavailable"),
        ),
        patch("neuralls.platform.tracking.failure_reporting.logger.warning") as warning,
    ):
        MLflowWorkflowFailureSink().record(context, _captured_error())

    warning.assert_called_once()


def test_mlflow_failure_sink_without_an_active_run_writes_nothing() -> None:
    context = ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.REPORTING,
        case_config_path=Path("case.toml"),
    )
    with (
        patch(f"{_SINK}.mlflow.active_run", return_value=None),
        patch(f"{_SINK}.mlflow.log_dict") as log_dict,
        patch(f"{_SINK}.mlflow.log_text") as log_text,
        patch(f"{_SINK}.mlflow.set_tags") as set_tags,
    ):
        MLflowWorkflowFailureSink().record(context, _captured_error())

    assert (log_dict.call_count, log_text.call_count, set_tags.call_count) == (0, 0, 0)
