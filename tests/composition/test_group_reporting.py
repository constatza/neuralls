"""Task 3 boundary for group reporting before method workloads exist."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from loguru import logger

from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.presentation import GroupPresentation


@dataclass(slots=True)
class RecordingSink:
    """Record any aggregate that incorrectly crosses the deferred boundary."""

    outcomes: list[GroupAggregated] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del presentation
        self.outcomes.append(outcome)


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(handler_id)


def test_system_major_group_reporting_defers_without_fabricating_events(
    warnings_log: list[str],
) -> None:
    sink = RecordingSink()

    report_all_levels((), (), {}, {}, sink)

    assert sink.outcomes == []
    assert any("canonical method workloads" in message for message in warnings_log)
