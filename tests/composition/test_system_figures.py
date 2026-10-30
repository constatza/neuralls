"""Per-system convergence figures draw every method on one system, one figure per metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from neuralls.composition.comparison import system_figures
from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.composition.comparison.system_figures import (
    concat_convergence_frames,
    write_system_figures,
)
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    MethodPresentationCandidate,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models import MeasuredPeakMemory, SystemTag
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.figure_text import EditorialTitle
from neuralls.platform.reporting.plots import ConvergenceCurve
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SystemId
from tests.convergence_support import cg_result
from tests.recipe_support import GAUSSIAN

METHODS = ("alpha", "beta", "gamma")
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
) -> tuple[CompletedMethodWorkload, ...]:
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
    workloads: list[CompletedMethodWorkload] = []
    for prepared in cohort.methods:
        diagnostics: list[tuple[SystemId, CGComparisonResult]] = [
            (tag.system_id, cg_result([1.0, 0.1, 0.01], energy_by_system[str(tag.system_id)]))
            for tag in systems
        ]
        workloads.append(
            CompletedMethodWorkload(
                method=prepared.method_id,
                planned=systems,
                setup_events=(),
                solve_attempts=(),
                diagnostics=tuple(diagnostics),
                generation_cost=None,
                workload_peak_memory=MeasuredPeakMemory(1),
                config_digest="sha256:" + "a" * 64,
                method_config={"name": str(prepared.method_id)},
                presentation=ExecutedMethod(prepared, ()),
                plot_styles={
                    str(prepared.method_id): PreconditionerPlotStyle(
                        prepared, PreconditionerType.AMG
                    )
                },
            )
        )
    return tuple(workloads)


def _draw(
    workloads: tuple[CompletedMethodWorkload, ...],
    dataset_presentation: DatasetPresentation,
    notation: NotationPolicy,
) -> None:
    write_system_figures(
        workloads,
        concat_convergence_frames(workloads),
        datasets={DatasetId("ds"): dataset_presentation},
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
    workloads = _workloads(systems, dataset_presentation, {"sys-1": energy, "sys-2": energy})

    _draw(workloads, dataset_presentation, notation)

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
    workloads = _workloads(
        systems, dataset_presentation, {"sys-1": [1.0, 0.2, 0.02], "sys-2": None}
    )

    _draw(workloads, dataset_presentation, notation)

    assert [r.title for r in draws.requests] == [
        "Residual History (matrix 1, RHS 0)",
        "Energy History (matrix 1, RHS 0)",
        "Residual History (matrix 2, RHS 0)",
    ]
