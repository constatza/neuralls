"""What each group figure variant summarizes, shared by the registry, table and card tests."""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.solver.reporting.metrics import MetricId

VARIANT_METRIC: Mapping[str, MetricId] = {
    "ecdf_iterations": MetricId.ITERATIONS,
    "ecdf_time_solve": MetricId.SOLVE_S,
    "ecdf_time_online": MetricId.ONLINE_S,
    "performance_profile_solve": MetricId.SOLVE_S,
    "performance_profile_online": MetricId.ONLINE_S,
    "convergence_factor": MetricId.CONVERGENCE_FACTOR,
}
"""The per-system metric of each variant whose figure draws one metric of the observation view."""
