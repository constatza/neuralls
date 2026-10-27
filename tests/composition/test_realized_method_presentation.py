"""Realized scientific facts are captured once per matrix and survive method-major execution."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from torchalg.preconditioners.base import Preconditioner

from neuralls.composition.comparison import _preconditioner_setup, comparison_run
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    MatrixWorkloadMember,
    ResolvedMethod,
)
from neuralls.domain.presentation.methods import (
    DuplicateMethodPresentationError,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    RealizedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.presentation.planner import realize_method
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.catalog import REALIZED_LEVELS
from neuralls.shared.scientific_description.models import (
    FieldImportance,
    IntegerFact,
    ScientificFact,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId
from tests.presentation_support import PreparedMethodFactory


def _levels(depth: int) -> tuple[ScientificFact, ...]:
    return (IntegerFact(REALIZED_LEVELS, depth, FieldImportance.ESSENTIAL),)


class _Built:
    """A weak-referenceable stand-in for a heavy built preconditioner."""


@dataclass(slots=True)
class _Counters:
    setups: int = 0
    captures: int = 0
    built: list[weakref.ref[_Built]] = field(default_factory=list)


@pytest.fixture
def solver_params() -> SolverParams:
    return SolverParams(
        rtol=1e-6, atol=1e-10, max_iterations=20, stopping_criterion="residual_norm", m_max=5
    )


@pytest.fixture
def matrix_bundles(tmp_path: Path) -> tuple[MatrixWorkload, MatrixWorkload]:
    bundles = []
    for matrix_id in (1, 2):
        members = tuple(
            MatrixWorkloadMember(
                tag=SystemTag(
                    DatasetId("ds"),
                    "gaussian",
                    MatrixId(matrix_id),
                    ordinal,
                    SystemId(f"ds-{matrix_id}-{ordinal}"),
                ),
                rhs=np.array([1.0, float(ordinal + 1)]),
            )
            for ordinal in (0, 1)
        )
        bundles.append(
            MatrixWorkload(
                dataset=DatasetId("ds"),
                matrix_id=MatrixId(matrix_id),
                matrix=np.diag(np.array([2.0, 3.0])) * matrix_id,
                matrix_path=tmp_path / f"matrix-{matrix_id}",
                matrix_index=matrix_id,
                members=members,
            )
        )
    return bundles[0], bundles[1]


@pytest.fixture
def method() -> ResolvedMethod:
    return ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.JACOBI))


@pytest.fixture
def counters(monkeypatch: pytest.MonkeyPatch, method: ResolvedMethod) -> _Counters:
    """Count setup builds and realized-metadata captures; solving is replaced by a stub."""
    counts = _Counters()

    def build(*args: object, **kwargs: object) -> dict[str, object]:
        counts.setups += 1
        built = _Built()
        counts.built.append(weakref.ref(built))
        return {str(method.name): built}

    def capture(precond: Preconditioner) -> tuple[ScientificFact, ...]:
        counts.captures += 1
        return _levels(2)

    def solve(
        matrix: torch.Tensor, rhs: torch.Tensor, *, preconditioners: dict[str, object], **_: object
    ) -> dict[str, CGComparisonResult]:
        name = next(iter(preconditioners))
        return {
            name: CGComparisonResult(
                x=np.ones(2),
                converged=True,
                iterations=3,
                residual=1e-8,
                residual_abs=1e-8,
                residual_history_rel=[1.0],
                residual_history_abs=[1.0],
                preconditioner=name,
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
                solve_time_seconds=1.0,
                solve_peak_memory_bytes=8,
                solve_time_samples=(1.0,),
            )
        }

    monkeypatch.setattr(comparison_run, "_build_scheduled", build)
    monkeypatch.setattr(_preconditioner_setup, "describe_realized_preconditioner", capture)
    monkeypatch.setattr(comparison_run, "run_cg_comparison", solve)
    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *a, **k: None)
    return counts


def _execute(
    method: ResolvedMethod,
    bundles: tuple[MatrixWorkload, ...],
    prepared: PreparedMethodPresentation,
    params: SolverParams,
    settings: NeurallsSettings,
) -> CompletedMethodWorkload:
    return comparison_run.execute_method_workload(
        method,
        bundles,
        prepared=prepared,
        params=params,
        settings=settings,
        notation_policy=NotationPolicy.default(),
    )


def test_facts_are_captured_once_per_realized_matrix_with_one_setup_each(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
) -> None:
    prepared = prepared_method(str(method.name))

    workload = _execute(method, matrix_bundles, prepared, solver_params, neuralls_settings)

    assert counters.setups == 2
    assert counters.captures == 2
    assert isinstance(workload.presentation, RealizedMethodPresentation)
    assert workload.presentation.prepared is prepared
    assert [(s.system_id, s.matrix_id) for s in workload.presentation.systems] == [
        (SystemId("ds-1-0"), MatrixId(1)),
        (SystemId("ds-1-1"), MatrixId(1)),
        (SystemId("ds-2-0"), MatrixId(2)),
        (SystemId("ds-2-1"), MatrixId(2)),
    ]
    assert all(s.facts == _levels(2) for s in workload.presentation.systems)


def test_built_preconditioners_are_released_after_capture(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
) -> None:
    workload = _execute(
        method,
        matrix_bundles,
        prepared_method(str(method.name)),
        solver_params,
        neuralls_settings,
    )
    gc.collect()

    assert len(counters.built) == 2
    assert all(ref() is None for ref in counters.built)
    assert workload.presentation is not None


def test_reporting_neither_sets_up_nor_captures_after_release(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
) -> None:
    workload = _execute(
        method,
        matrix_bundles,
        prepared_method(str(method.name)),
        solver_params,
        neuralls_settings,
    )
    before = (counters.setups, counters.captures)
    sink = _RecordingSink()

    report_all_levels((workload,), sink)

    assert (counters.setups, counters.captures) == before
    assert sink.presentations


def test_failed_setup_is_reported_from_the_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("setup broke")

    monkeypatch.setattr(comparison_run, "_build_scheduled", fail)
    prepared = prepared_method(str(method.name))

    workload = _execute(method, matrix_bundles, prepared, solver_params, neuralls_settings)

    assert workload.presentation is prepared
    sink = _RecordingSink()
    report_all_levels((workload,), sink)
    group_method = sink.presentations[0].methods[method.name]
    assert group_method.prepared is prepared
    assert group_method.invariant_facts == ()


def test_a_method_with_unrelated_prepared_identity_is_rejected(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
) -> None:
    with pytest.raises(ValueError, match="prepared presentation"):
        _execute(method, matrix_bundles, prepared_method("other"), solver_params, neuralls_settings)


@dataclass(slots=True)
class _RecordingSink:
    presentations: list[GroupPresentation] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del outcome
        self.presentations.append(presentation)


def _workload(
    prepared: PreparedMethodPresentation,
    depth_by_matrix: dict[int, int],
    *,
    label: str = "Jacobi",
) -> CompletedMethodWorkload:
    """A completed workload with one system per matrix, realized at the given depths."""
    name = prepared.method_id
    tags = tuple(
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(m), 0, SystemId(f"{name}-{m}"))
        for m in depth_by_matrix
    )
    events = tuple(
        MatrixSetupEvent(
            SetupEventId(f"{name}:ds:{tag.matrix_id}"),
            tag.dataset,
            tag.matrix_id,
            name,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SetupCompleted(),
        )
        for tag in tags
    )
    attempts = tuple(
        SolveAttempt(
            event.event_id,
            tag.system_id,
            tag.dataset,
            tag.recipe,
            tag.matrix_id,
            tag.rhs_ordinal,
            name,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SolveConverged(1, 1e-8),
        )
        for tag, event in zip(tags, events, strict=True)
    )
    realized = realize_method(
        prepared,
        tuple(
            SystemRealization(
                tag.system_id, tag.matrix_id, _levels(depth_by_matrix[int(tag.matrix_id)])
            )
            for tag in tags
        ),
    )
    return CompletedMethodWorkload(
        method=name,
        planned=tags,
        setup_events=events,
        solve_attempts=attempts,
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(1),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": str(name)},
        presentation=realized,
        plot_styles={str(name): PreconditionerPlotStyle(label, PreconditionerType.JACOBI)},
    )


def test_group_reports_a_realized_fact_shared_by_every_matrix(
    prepared_method: PreparedMethodFactory,
) -> None:
    prepared = prepared_method("jacobi")
    sink = _RecordingSink()

    report_all_levels((_workload(prepared, {1: 2, 2: 2}),), sink)

    group_method = sink.presentations[0].methods[PreconditionerName("jacobi")]
    assert group_method.invariant_facts == _levels(2)
    assert _levels(2)[0] in group_method.compact_facts
    assert _levels(2)[0] in group_method.detailed_facts


def test_group_does_not_claim_a_realized_fact_that_varies_by_matrix(
    prepared_method: PreparedMethodFactory,
) -> None:
    prepared = prepared_method("jacobi")
    workload = _workload(prepared, {1: 2, 2: 3})
    sink = _RecordingSink()

    report_all_levels((workload,), sink)

    group_method = sink.presentations[0].methods[PreconditionerName("jacobi")]
    assert group_method.invariant_facts == ()
    assert all(fact.key != REALIZED_LEVELS for fact in group_method.compact_facts)
    assert isinstance(workload.presentation, RealizedMethodPresentation)
    assert [s.facts for s in workload.presentation.systems] == [_levels(2), _levels(3)]


def test_colliding_group_labels_raise_instead_of_appending_a_key(
    prepared_method: PreparedMethodFactory,
) -> None:
    first = _workload(prepared_method("jacobi_a"), {1: 2})
    second = _workload(prepared_method("jacobi_b"), {1: 2})

    with pytest.raises(DuplicateMethodPresentationError) as raised:
        report_all_levels((first, second), _RecordingSink())

    assert set(raised.value.method_ids) == {"jacobi_a", "jacobi_b"}


def test_cohort_returns_the_prepared_presentation_of_a_known_method(
    prepared_presentation: PreparedComparisonPresentation,
) -> None:
    assert prepared_presentation.method(PreconditionerName("jacobi")).method_id == "jacobi"
    with pytest.raises(KeyError):
        prepared_presentation.method(PreconditionerName("absent"))
