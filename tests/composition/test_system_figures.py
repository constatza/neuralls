"""Per-system convergence figures draw every method on one system, one figure per metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest

from neuralls.composition.comparison.postprocessing import system_figures
from neuralls.composition.comparison.postprocessing.report_inputs import ReportedMethod
from neuralls.composition.comparison.postprocessing.system_figures import (
    SystemFigure,
    SystemMetric,
    write_system_figures,
)
from neuralls.domain.benchmarking.selection import NoReferenceConfigured
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    MethodPresentationCandidate,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models import SystemTag
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.figure_scope import SystemScope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.reporting.event_frames import build_convergence_frame
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.figure_metadata import FigureKind
from neuralls.platform.reporting.figure_text import EditorialTitle
from neuralls.platform.reporting.plots import ConvergenceCurve
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SystemId
from tests.convergence_support import cg_result
from tests.recipe_support import GAUSSIAN

METHODS = ("alpha", "beta", "gamma")
CONTEXT = SolverTimingContext(
    rtol=1e-6,
    atol=1e-14,
    max_iterations=20,
    stopping_criterion="residual_norm",
    m_max=5,
    reference_precision_margin=2.0,
    timing_repeats=1,
    timing_warmup=0,
    timing_statistic="median",
    normalization="none",
)
FIGURE_ROOT = Path("figures")


@dataclass(slots=True)
class DrawRequest:
    curves: dict[str, ConvergenceCurve]
    path: Path
    title: str
    ylabel: str


@dataclass(slots=True)
class RecordedDraws:
    requests: list[DrawRequest] = field(default_factory=list)


@pytest.fixture
def draws(monkeypatch: pytest.MonkeyPatch) -> RecordedDraws:
    recorded = RecordedDraws()

    def record(
        curves: dict[str, ConvergenceCurve],
        *,
        save_path: Path,
        title: EditorialTitle,
        ylabel: str,
        **_: object,
    ) -> None:
        recorded.requests.append(DrawRequest(curves, Path(save_path), title.text, ylabel))

    monkeypatch.setattr(system_figures, "plot_convergence_curves", record)
    return recorded


@pytest.fixture
def systems() -> tuple[SystemTag, ...]:
    return tuple(
        SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(matrix), 0, SystemId(f"sys-{matrix}"))
        for matrix in (1, 2)
    )


def _workloads(
    systems: tuple[SystemTag, ...],
    dataset_presentation: DatasetPresentation,
    energy_by_system: dict[str, list[float] | None],
) -> tuple[tuple[ReportedMethod, ...], pl.DataFrame]:
    cohort = prepare_presentations(
        UnvalidatedComparisonPresentation(
            methods=tuple(
                MethodPresentationCandidate(
                    PreconditionerName(name),
                    StandaloneMethodPresentation(ScientificDescription(f"Method {name}")),
                )
                for name in METHODS
            ),
            scope=ComparisonScopePresentation(dataset_presentation),
        )
    )
    workloads: list[ReportedMethod] = []
    frames: list[pl.DataFrame] = []
    for prepared in cohort.methods:
        diagnostics: list[tuple[SystemId, CGComparisonResult]] = [
            (tag.system_id, cg_result([1.0, 0.1, 0.01], energy_by_system[str(tag.system_id)]))
            for tag in systems
        ]
        frames.append(build_convergence_frame(prepared.method_id, diagnostics))
        workloads.append(
            ReportedMethod(
                method=prepared.method_id,
                run_id=f"run-{prepared.method_id}",
                config_digest="sha256:" + "c" * 64,
                planned=systems,
                setup_events=(),
                solve_attempts=(),
                presentation=ExecutedMethod(prepared, ()),
                plot_styles={
                    str(prepared.method_id): PreconditionerPlotStyle(
                        prepared, PreconditionerType.AMG
                    )
                },
            )
        )
    return tuple(workloads), pl.concat(frames)


def _draw(
    workloads: tuple[ReportedMethod, ...],
    frame: pl.DataFrame,
    dataset_presentation: DatasetPresentation,
    notation: NotationPolicy,
) -> tuple[SystemFigure, ...]:
    tags = {tag.system_id: tag for workload in workloads for tag in workload.planned}
    return write_system_figures(
        workloads,
        frame,
        datasets={DatasetId("ds"): dataset_presentation},
        references={system_id: NoReferenceConfigured() for system_id in tags},
        solver_contexts={system_id: CONTEXT for system_id in tags},
        notation=notation,
        root=FIGURE_ROOT,
    )


def test_each_system_gets_one_residual_and_one_energy_figure_with_every_method(
    draws: RecordedDraws,
    systems: tuple[SystemTag, ...],
    dataset_presentation: DatasetPresentation,
    notation: NotationPolicy,
) -> None:
    energy = [1.0, 0.2, 0.02]
    workloads, frame = _workloads(systems, dataset_presentation, {"sys-1": energy, "sys-2": energy})

    _draw(workloads, frame, dataset_presentation, notation)

    assert [(r.title, tuple(r.curves)) for r in draws.requests] == [
        (f"{title} (matrix {matrix}, RHS 0)", METHODS)
        for matrix in (1, 2)
        for title in ("Residual History", "Energy History")
    ]
    names = [r.path.relative_to(FIGURE_ROOT).parts for r in draws.requests]
    assert {parts[1] for parts in names} == {"residual_history.png", "energy_history.png"}
    assert all("sys-" not in part and "ds" != part for parts in names for part in parts)
    assert len({parts[0] for parts in names}) == 2


def test_a_system_with_only_null_energy_skips_just_its_energy_figure(
    draws: RecordedDraws,
    systems: tuple[SystemTag, ...],
    dataset_presentation: DatasetPresentation,
    notation: NotationPolicy,
) -> None:
    workloads, frame = _workloads(
        systems, dataset_presentation, {"sys-1": [1.0, 0.2, 0.02], "sys-2": None}
    )

    _draw(workloads, frame, dataset_presentation, notation)

    assert [r.title for r in draws.requests] == [
        "Residual History (matrix 1, RHS 0)",
        "Energy History (matrix 1, RHS 0)",
        "Residual History (matrix 2, RHS 0)",
    ]


def test_each_figure_is_planned_with_its_system_methods_parameters_and_sources(
    draws: RecordedDraws,
    systems: tuple[SystemTag, ...],
    dataset_presentation: DatasetPresentation,
    notation: NotationPolicy,
) -> None:
    energy = [1.0, 0.2, 0.02]
    workloads, frame = _workloads(systems, dataset_presentation, {"sys-1": energy, "sys-2": energy})

    figures = _draw(workloads, frame, dataset_presentation, notation)

    second_residual = next(
        figure
        for figure in figures
        if figure.tag.matrix_id == MatrixId(2) and figure.metric is SystemMetric.RESIDUAL
    )
    plan = second_residual.staged.plan
    metadata = plan.metadata
    assert plan.filename == "residual_history.png"
    assert second_residual.staged.path.name == plan.filename
    assert metadata.kind is FigureKind.RESIDUAL_HISTORY
    assert metadata.report_level is ReportLevelName.SYSTEM
    assert metadata.scope == SystemScope(dataset_presentation, GAUSSIAN, MatrixId(2), 0)
    assert (metadata.n_matrices, metadata.n_planned_rhs) == (1, 1)
    assert metadata.solver_contexts == (CONTEXT,)
    assert [method.method_id for method in metadata.methods] == list(METHODS)
    assert [source.run_id for source in metadata.sources] == [f"run-{name}" for name in METHODS]
    assert metadata.sources[0].artifact_paths == (
        "method_workload/convergence.parquet",
        "method_workload/solve_attempts.parquet",
    )
