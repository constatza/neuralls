"""Builders of the system provenance and dataset presentations a method workload persists."""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.shared.types import DatasetId

SOLVER_CONTEXT = SolverTimingContext(
    rtol=1e-6,
    atol=1e-10,
    max_iterations=20,
    stopping_criterion="residual_norm",
    m_max=5,
    reference_precision_margin=10.0,
    timing_repeats=3,
    timing_warmup=1,
    timing_statistic="median",
    normalization="matrix",
)


def provenance_for(
    planned: tuple[SystemTag, ...],
    reference: ReferenceSelection | None = None,
) -> tuple[SystemProvenance, ...]:
    """Provenance of every planned system under one solver context and reference."""
    return tuple(
        SystemProvenance(tag, SOLVER_CONTEXT, reference or NoReferenceConfigured())
        for tag in planned
    )


def datasets_for(
    planned: tuple[SystemTag, ...], presentation: DatasetPresentation
) -> Mapping[DatasetId, DatasetPresentation]:
    """The one dataset presentation for every dataset a workload ran on."""
    return {tag.dataset: presentation for tag in planned}
