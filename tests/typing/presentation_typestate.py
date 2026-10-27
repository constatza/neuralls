"""Static fixture: execution consumes the prepared state, detailed reporting the executed one.

Checked by ``ty``; it is never collected or run by pytest.
"""

from __future__ import annotations

from typing import assert_type

from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    PreparedComparisonPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations


def execute(cohort: PreparedComparisonPresentation) -> None:
    """Stand-in for execution and tracking entry points."""


def report_detailed(method: ExecutedMethod) -> None:
    """Stand-in for per-system detailed reporting."""


def lifecycle(
    unvalidated: UnvalidatedComparisonPresentation, systems: tuple[SystemRealization, ...]
) -> None:
    """Walk the valid unvalidated -> prepared -> executed path."""
    prepared = prepare_presentations(unvalidated)
    assert_type(prepared, PreparedComparisonPresentation)
    execute(prepared)

    realized = ExecutedMethod(prepared.methods[0], systems)
    assert_type(realized, ExecutedMethod)
    report_detailed(realized)
