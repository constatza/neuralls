"""The typed inputs a report draws from, rebuilt from a compare session's stored artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    SolveAttempt,
    SystemTag,
)
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    decode_setup_frame,
    decode_solve_frame,
)
from neuralls.platform.config.models.preconditioner_family import (
    pod2g_style_keys,
    preconditioner_family,
)
from neuralls.platform.reporting.stored_benchmark import StoredBenchmark, StoredMethod
from neuralls.shared.types import DatasetId, PreconditionerName, SystemId


@dataclass(frozen=True, slots=True)
class ReportedMethod:
    """One method's events and presentation as the report levels consume them."""

    method: PreconditionerName
    run_id: str
    config_digest: str
    planned: tuple[SystemTag, ...]
    setup_events: tuple[MatrixSetupEvent, ...]
    solve_attempts: tuple[SolveAttempt, ...]
    presentation: ExecutedMethod
    plot_styles: Mapping[str, PreconditionerPlotStyle]


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """Everything a report needs from one compare session."""

    stored: StoredBenchmark
    methods: tuple[ReportedMethod, ...]
    datasets: Mapping[DatasetId, DatasetPresentation]
    references: Mapping[SystemId, ReferenceSelection]
    solver_contexts: Mapping[SystemId, SolverTimingContext]
    convergence: pl.DataFrame


def _reported_method(stored: StoredMethod) -> ReportedMethod:
    color_key, marker_key = pod2g_style_keys(stored.config)
    return ReportedMethod(
        method=stored.name,
        run_id=stored.run_id,
        config_digest=stored.config_digest,
        planned=tuple(system.tag for system in stored.systems),
        setup_events=decode_setup_frame(stored.setup_events),
        solve_attempts=decode_solve_frame(stored.solve_attempts),
        presentation=stored.presentation,
        plot_styles={
            str(stored.name): PreconditionerPlotStyle(
                presentation=stored.presentation.prepared,
                family=preconditioner_family(stored.config),
                color_key=color_key,
                marker_key=marker_key,
            )
        },
    )


def concat_convergence(methods: Sequence[StoredMethod]) -> pl.DataFrame:
    """The convergence rows of every stored method as one table."""
    frames = [method.convergence for method in methods]
    return pl.concat(frames) if frames else pl.DataFrame(schema=CONVERGENCE_SCHEMA)


def report_inputs(stored: StoredBenchmark) -> ReportInputs:
    """Rebuild typed events, references and solver contexts from a validated stored benchmark."""
    systems = [system for method in stored.methods for system in method.systems]
    return ReportInputs(
        stored=stored,
        methods=tuple(_reported_method(method) for method in stored.methods),
        datasets=stored.datasets,
        references={system.tag.system_id: system.reference for system in systems},
        solver_contexts={system.tag.system_id: system.solver for system in systems},
        convergence=concat_convergence(stored.methods),
    )
