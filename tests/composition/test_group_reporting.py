"""Aggregate reporting consumes only completed canonical method workloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)


@dataclass(slots=True)
class RecordingSink:
    outcomes: list[GroupAggregated] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del presentation
        self.outcomes.append(outcome)


def test_group_reporting_reduces_completed_workload_events() -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("system"))
    event_id = SetupEventId("identity:ds:1")
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(
            MatrixSetupEvent(
                event_id,
                tag.dataset,
                tag.matrix_id,
                method,
                1.0,
                (1.0,),
                MeasuredPeakMemory(10),
                SetupCompleted(),
            ),
        ),
        solve_attempts=(
            SolveAttempt(
                event_id,
                tag.system_id,
                tag.dataset,
                tag.recipe,
                tag.matrix_id,
                tag.rhs_ordinal,
                method,
                2.0,
                (2.0,),
                MeasuredPeakMemory(20),
                SolveConverged(3, 1e-8),
            ),
        ),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(20),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": "identity"},
    )
    sink = RecordingSink()

    report_all_levels((workload,), sink)

    assert [outcome.n_planned_rhs for outcome in sink.outcomes] == [1, 1, 1]
