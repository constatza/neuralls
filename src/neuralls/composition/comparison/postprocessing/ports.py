"""The consumer-side contract a report writes through; it knows scopes, never MLflow runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport, WorkloadMixtureReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.report_scope import System
from neuralls.platform.reporting.figure_metadata import StagedFigure


class ReportSink(Protocol):
    """Receives each aggregation level of a report and records it under its own scope."""

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Record the metrics and figures of one aggregated group at the group's scope."""
        ...

    def report_system(
        self, scope: System, metrics: Mapping[str, float], figures: Sequence[StagedFigure]
    ) -> None:
        """Record one system's per-method metrics and its planned figures at its own scope."""
        ...

    def finalize(self) -> None:
        """Flush report-wide artifacts once every scope has been recorded."""
        ...
