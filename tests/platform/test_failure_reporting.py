"""Blocking workflow failures retain exact context in the run of the scope that failed."""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest
from loguru import logger
from tests.fakes.run_writer import RecordingRunWriter

from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.shared.types import PreconditionerName
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
    ComparisonMethodFailureContext,
    ComparisonMethodPhase,
)

METHOD_RUN = "method-run"


class _UnavailableStore(RecordingRunWriter):
    def log_dict(self, run_id: str, data: object, artifact_file: str) -> None:
        raise RuntimeError("tracking unavailable")


@pytest.fixture
def captured_error() -> RuntimeError:
    try:
        raise RuntimeError("injected workload failure")
    except RuntimeError as exc:
        return exc


@pytest.fixture
def method_failure() -> ComparisonMethodFailureContext:
    return ComparisonMethodFailureContext(
        phase=ComparisonMethodPhase.EXECUTION,
        method=PreconditionerName("amg"),
        config_digest="sha256:" + "1" * 64,
        system_ids=("system-1", "system-2"),
    )


def test_mlflow_failure_sink_records_exact_context_and_traceback_into_its_run(
    method_failure: ComparisonMethodFailureContext, captured_error: RuntimeError
) -> None:
    writer = RecordingRunWriter()

    MLflowWorkflowFailureSink(writer, METHOD_RUN).record(method_failure, captured_error)

    (run_id,) = writer.runs
    run = writer.runs[run_id]
    assert run_id == METHOD_RUN
    assert run.dicts == {
        "failure/context.json": {
            "phase": "execution",
            "exception_type": "RuntimeError",
            "message": "injected workload failure",
            "scope": "method",
            "method": "amg",
            "config_digest": method_failure.config_digest,
            "system_ids": ["system-1", "system-2"],
        }
    }
    assert run.texts == {
        "failure/traceback.txt": "".join(traceback.format_exception(captured_error))
    }
    assert run.tags == {
        "failure.phase": "execution",
        "failure.scope": "method",
        "failure.exception_type": "RuntimeError",
        "failure.method": "amg",
    }


def test_mlflow_failure_sink_diagnostic_error_is_best_effort(
    captured_error: RuntimeError,
) -> None:
    context = ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.REPORTING,
        case_config_path=Path("case.toml"),
    )
    warnings: list[str] = []
    handler = logger.add(warnings.append, level="WARNING", format="{level}")
    try:
        MLflowWorkflowFailureSink(_UnavailableStore(), METHOD_RUN).record(context, captured_error)
    finally:
        logger.remove(handler)

    assert warnings == ["WARNING\n"]
