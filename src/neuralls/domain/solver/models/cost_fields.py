"""Scalar costs of one RHS diagnostic result, distinct from workload totals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from neuralls.domain.solver.models.cg_result import CGComparisonResult


@dataclass(frozen=True, slots=True)
class CostFieldSpec:
    """One reportable scalar cost, shared by every consumer that reports costs.

    `field_name` is the scalar's key on one result. `setup_time_seconds` is the
    full setup of the result's matrix, copied to each sibling RHS result; workload
    totals come from the setup and solve events (`reporting/aggregation.py`), which
    count each matrix's setup once.
    """

    field_name: str
    extractor: Callable[[CGComparisonResult], float | int | None]


COST_FIELD_REGISTRY: tuple[CostFieldSpec, ...] = (
    CostFieldSpec(
        field_name="offline_time_seconds",
        extractor=lambda r: r.offline_time_seconds,
    ),
    CostFieldSpec(
        field_name="setup_time_seconds",
        extractor=lambda r: r.setup_cost,
    ),
    CostFieldSpec(
        field_name="solve_time_seconds",
        extractor=lambda r: r.solve_time_seconds,
    ),
    CostFieldSpec(
        field_name="avg_iteration_time_seconds",
        extractor=lambda r: r.avg_iteration_time_seconds,
    ),
    CostFieldSpec(
        field_name="peak_memory_bytes",
        extractor=lambda r: r.peak_memory_bytes,
    ),
)
"""The scalar vocabulary of a single RHS result, not a workload reduction."""
