"""Static fixture: execution consumes the prepared state, detailed reporting the realized one.

Checked by ``ty``; it is never collected or run by pytest.
"""

from __future__ import annotations

from typing import assert_type

from neuralls.domain.presentation.methods import (
    PreparedComparisonPresentation,
    RealizedMethodPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations, realize_method


def execute(cohort: PreparedComparisonPresentation) -> None:
    """Stand-in for execution and tracking entry points."""


def report_detailed(method: RealizedMethodPresentation) -> None:
    """Stand-in for per-system detailed reporting."""


def lifecycle(
    unvalidated: UnvalidatedComparisonPresentation, systems: tuple[SystemRealization, ...]
) -> None:
    """Walk the valid unvalidated -> prepared -> realized path."""
    prepared = prepare_presentations(unvalidated)
    assert_type(prepared, PreparedComparisonPresentation)
    execute(prepared)

    realized = realize_method(prepared.methods[0], systems)
    assert_type(realized, RealizedMethodPresentation)
    report_detailed(realized)
