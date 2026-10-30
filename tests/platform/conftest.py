"""Fixtures shared by the platform storage tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import datasets_for, provenance_for
from tests.recipe_support import GAUSSIAN

from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    canonical_method_config,
)
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.benchmark_events import write_method_benchmark_artifacts
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId

type WriteMethod = Callable[..., CompletedMethodWorkload]


@pytest.fixture
def write_method(
    prepared_method: PreparedMethodFactory, dataset_presentation: DatasetPresentation
) -> WriteMethod:
    """Write a complete one-system artifact set for a standard preconditioner."""

    def write(
        kind: PreconditionerType, root: Path, *, dataset: DatasetPresentation = dataset_presentation
    ) -> CompletedMethodWorkload:
        config = StandardPreconditionerConfig(type=kind)
        method = PreconditionerName(config.name)
        tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(0), 0, SystemId(f"{method}-0"))
        setup = MatrixSetupEvent(
            SetupEventId(f"{method}-setup"),
            tag.dataset,
            tag.matrix_id,
            method,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SetupCompleted(),
        )
        attempt = SolveAttempt(
            setup.event_id,
            tag.system_id,
            tag.dataset,
            GAUSSIAN,
            tag.matrix_id,
            0,
            method,
            2.0,
            (2.0,),
            MeasuredPeakMemory(1),
            SolveConverged(2, 1e-8),
        )
        workload = CompletedMethodWorkload(
            method=method,
            planned=(tag,),
            setup_events=(setup,),
            solve_attempts=(attempt,),
            diagnostics=(),
            generation_cost=None,
            workload_peak_memory=MeasuredPeakMemory(2),
            config_digest="sha256:" + "a" * 64,
            method_config=canonical_method_config(config),
            presentation=ExecutedMethod(prepared_method(str(method)), ()),
        )
        summary = summarize_method_workload(
            method, workload.planned, (setup,), (attempt,), workload.workload_peak_memory
        )
        write_method_benchmark_artifacts(
            workload,
            summary,
            root,
            systems=provenance_for(workload.planned),
            datasets=datasets_for(workload.planned, dataset),
        )
        return workload

    return write
