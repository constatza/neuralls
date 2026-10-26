"""Typed locations for blocking comparison-workflow failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from neuralls.shared.digest import Digest
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SystemId


class ComparisonBatchPhase(StrEnum):
    """Blocking phases owned by the comparison session parent."""

    WARMUP = "warmup"
    METHOD_WORKLOADS = "method_workloads"
    RESULT_PROJECTION = "result_projection"
    REPORTING = "reporting"
    FINALIZATION = "finalization"


class ComparisonMethodPhase(StrEnum):
    """Blocking phases owned by one method child run."""

    EXECUTION = "execution"
    PERSISTENCE = "persistence"


class ComparisonMatrixPhase(StrEnum):
    """Blocking phases tied to one method/matrix pair."""

    INPUT_LOADING = "input_loading"


@dataclass(frozen=True, slots=True)
class ComparisonBatchFailureContext:
    """Location of a session-wide blocking failure."""

    phase: ComparisonBatchPhase
    case_config_path: Path


@dataclass(frozen=True, slots=True)
class ComparisonMethodFailureContext:
    """Location of a blocking failure inside one method child."""

    phase: ComparisonMethodPhase
    method: PreconditionerName
    config_digest: Digest
    system_ids: tuple[SystemId, ...]


@dataclass(frozen=True, slots=True)
class ComparisonMatrixFailureContext:
    """Location of a blocking failure while preparing one matrix workload."""

    phase: ComparisonMatrixPhase
    method: PreconditionerName
    dataset: DatasetId
    matrix_id: MatrixId


type WorkflowFailureContext = (
    ComparisonBatchFailureContext | ComparisonMethodFailureContext | ComparisonMatrixFailureContext
)


class WorkflowFailureSink(Protocol):
    """Records a blocking failure without owning exception transport."""

    def record(self, context: WorkflowFailureContext, exc: Exception) -> None:
        """Record the failure in the current infrastructure context."""
        ...
