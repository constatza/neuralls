"""The scalar costs reported per preconditioner per comparison run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from neuralls.domain.solver.models.cg_result import CGComparisonResult


@dataclass(frozen=True, slots=True)
class CostFieldSpec:
    """One reportable scalar cost, shared by every consumer that reports costs.

    `field_name` is the stable key written to `comparison.toml`; `extractor`
    is the same accessor the cost bar-charts use. Titles and filenames are
    presentation and live with the plots (`composition/comparison/_plots.py`),
    keyed by `field_name` — a test keeps the two in step.
    """

    field_name: str
    extractor: Callable[[CGComparisonResult], float | int | None]


COST_FIELD_REGISTRY: tuple[CostFieldSpec, ...] = (
    CostFieldSpec(
        field_name="generation_time_seconds",
        extractor=lambda r: r.offline_generation_time_seconds,
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
"""Every scalar cost reported per preconditioner per comparison run.

Iterated by `_plots.py`'s bar-chart generation.
"""
