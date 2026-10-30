"""Projection of completed method workloads onto per-comparison outcomes (in memory)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from neuralls.composition.assignments.comparison_batch import (
    PreparedComparisonExecution,
    _outcomes_from_workloads,
)
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonEntryFailed,
    ComparisonPresentation,
    CompletedMethodWorkload,
)
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import SystemId
from tests.convergence_support import cg_result
from tests.presentation_support import PreparedMethodFactory
from tests.realized_support import completed_workload

SYSTEM = "jacobi-0"
SUMMARY = (
    "CG comparison results:\n"
    "- Jacobi             status=ok   iters=  1  rel_res=1.000e-08 (abs=1.000e-08)"
)


@pytest.fixture
def workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    """One completed Jacobi workload whose single system converged in one iteration."""
    return replace(
        completed_workload(prepared_method("jacobi"), {0: 1}),
        diagnostics=((SystemId(SYSTEM), cg_result([1e-8])),),
    )


def _pending(
    entry_id: str,
    methods: PreparedComparisonPresentation,
    comparison_run: NamedRun,
) -> MagicMock:
    item = MagicMock(spec=PreparedComparisonExecution)
    item.entry = MagicMock()
    item.entry.id = entry_id
    item.resolved = MagicMock()
    item.resolved.warnings = ()
    item.run_name = comparison_run
    item.comparison_presentation = ComparisonPresentation(
        methods=methods, title=MagicMock(), notation=NotationPolicy.default()
    )
    return item


def test_completed_item_carries_the_formatted_summary(
    workload: CompletedMethodWorkload,
    prepared_method: PreparedMethodFactory,
    dataset_presentation: DatasetPresentation,
    comparison_run: NamedRun,
) -> None:
    cohort = PreparedComparisonPresentation(
        (prepared_method("jacobi"),), ComparisonScopePresentation(dataset_presentation)
    )

    [outcome] = _outcomes_from_workloads([_pending(SYSTEM, cohort, comparison_run)], [workload])

    assert outcome == ComparisonCompleted(comparison_id=SYSTEM, run=comparison_run, summary=SUMMARY)


def test_missing_method_presentation_yields_entry_failed(
    workload: CompletedMethodWorkload,
    prepared_method: PreparedMethodFactory,
    dataset_presentation: DatasetPresentation,
    comparison_run: NamedRun,
) -> None:
    cohort = PreparedComparisonPresentation(
        (prepared_method("identity"),), ComparisonScopePresentation(dataset_presentation)
    )

    [outcome] = _outcomes_from_workloads([_pending(SYSTEM, cohort, comparison_run)], [workload])

    assert isinstance(outcome, ComparisonEntryFailed)
    assert outcome.comparison_id == SYSTEM
