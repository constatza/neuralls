"""What each group figure variant summarizes, shared by the registry, table and card tests."""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.solver.models.cost import NO_OFFLINE_COST, OfflineStages
from neuralls.domain.solver.models.group_outcomes import GroupReport
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import PreconditionerName

VARIANT_METRIC: Mapping[str, MetricId] = {
    "ecdf_iterations": MetricId.ITERATIONS,
    "ecdf_time_solve": MetricId.SOLVE_S,
    "ecdf_time_online": MetricId.ONLINE_S,
    "performance_profile_solve": MetricId.SOLVE_S,
    "performance_profile_online": MetricId.ONLINE_S,
    "convergence_factor": MetricId.CONVERGENCE_FACTOR,
}
"""The per-system metric of each variant whose figure draws one metric of the observation view."""


def no_offline_cost(report: GroupReport) -> dict[PreconditionerName, OfflineStages]:
    """Every method of ``report`` has nothing to generate or fit."""
    return {summary.method: NO_OFFLINE_COST for summary in report.summaries}
