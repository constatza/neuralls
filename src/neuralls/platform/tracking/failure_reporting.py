"""MLflow adapter for blocking comparison-workflow diagnostics."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import assert_never

from loguru import logger

from neuralls.platform.tracking.run_writer import RunWriter
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonMatrixFailureContext,
    ComparisonMethodFailureContext,
    WorkflowFailureContext,
)

FAILURE_CONTEXT_ARTIFACT = "failure/context.json"
FAILURE_TRACEBACK_ARTIFACT = "failure/traceback.txt"


@dataclass(frozen=True, slots=True)
class MLflowWorkflowFailureSink:
    """Persist exact failure context and traceback into the run of the scope that failed."""

    writer: RunWriter
    run_id: str

    def record(self, context: WorkflowFailureContext, exc: Exception) -> None:
        """Best-effort recording that never replaces the workflow exception."""
        try:
            record, tags = _failure_record(context, exc)
            self.writer.log_dict(self.run_id, record, FAILURE_CONTEXT_ARTIFACT)
            self.writer.log_text(
                self.run_id,
                "".join(traceback.format_exception(exc)),
                FAILURE_TRACEBACK_ARTIFACT,
            )
            self.writer.set_tags(self.run_id, tags)
        except Exception as diagnostic_exc:  # noqa: BLE001
            logger.warning(
                "Failed to record blocking workflow failure '{}': {}. Original failure: {}",
                type(exc).__name__,
                diagnostic_exc,
                exc,
            )


def _failure_record(
    context: WorkflowFailureContext, exc: Exception
) -> tuple[dict[str, object], dict[str, str]]:
    common: dict[str, object] = {
        "phase": context.phase.value,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    tags = {
        "failure.phase": context.phase.value,
        "failure.exception_type": type(exc).__name__,
    }
    match context:
        case ComparisonBatchFailureContext(case_config_path=case_config_path):
            record = {**common, "scope": "batch", "case_config_path": str(case_config_path)}
            return record, {**tags, "failure.scope": "batch"}
        case ComparisonMethodFailureContext(
            method=method, config_digest=config_digest, system_ids=system_ids
        ):
            record = {
                **common,
                "scope": "method",
                "method": str(method),
                "config_digest": str(config_digest),
                "system_ids": [str(system_id) for system_id in system_ids],
            }
            return record, {
                **tags,
                "failure.scope": "method",
                "failure.method": str(method),
            }
        case ComparisonMatrixFailureContext(method=method, dataset=dataset, matrix_id=matrix_id):
            record = {
                **common,
                "scope": "matrix",
                "method": str(method),
                "dataset": str(dataset),
                "matrix_id": int(matrix_id),
            }
            return record, {
                **tags,
                "failure.scope": "matrix",
                "failure.method": str(method),
                "failure.dataset": str(dataset),
                "failure.matrix_id": str(int(matrix_id)),
            }
        case unreachable:
            assert_never(unreachable)
