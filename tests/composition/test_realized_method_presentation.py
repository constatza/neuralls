"""Realized scientific facts are captured once per matrix and survive method-major execution."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torchalg.preconditioners.base import Preconditioner

from neuralls.composition.comparison import _preconditioner_setup, comparison_run
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    RealizedFactCaptureError,
)
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    MatrixWorkloadMember,
    ResolvedMethod,
)
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    MissingMethodPresentationError,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.figure_scope import AllDatasetsScope, DatasetScope, RecipeScope
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.platform.config.models.preconditioner import (
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.catalog import REALIZED_LEVELS
from neuralls.shared.scientific_description.models import (
    ScientificFact,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SystemId
from neuralls.shared.workflow_failures import ComparisonMatrixFailureContext, ComparisonMatrixPhase
from tests.presentation_support import PreparedMethodFactory
from tests.realized_support import completed_workload, levels
from tests.recipe_support import ALPHA, BETA, GAUSSIAN


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
def matrix_bundles() -> tuple[MatrixWorkload, MatrixWorkload]:
    bundles = []
    for matrix_id in (1, 2):
        members = tuple(
            MatrixWorkloadMember(
                tag=SystemTag(
                    DatasetId("ds"),
                    GAUSSIAN,
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
                matrix_path=Path("/workspace") / f"matrix-{matrix_id}",
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
        return levels(2)

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


type Executor = Callable[
    [
        ResolvedMethod,
        tuple[MatrixWorkload, ...],
        PreparedMethodPresentation,
        SolverParams,
        NeurallsSettings,
    ],
    CompletedMethodWorkload,
]


@pytest.fixture
def execute(group_datasets: dict[DatasetId, DatasetPresentation]) -> Executor:
    """Execute one method over bundles, with log text drawn from the shared dataset fixture."""

    def run(
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
            datasets=group_datasets,
            notation=NotationPolicy.default(),
            params=params,
            settings=settings,
        )

    return run


def test_facts_are_captured_once_per_realized_matrix_with_one_setup_each(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    execute: Executor,
) -> None:
    prepared = prepared_method(str(method.name))

    workload = execute(method, matrix_bundles, prepared, solver_params, neuralls_settings)

    assert counters.setups == 2
    assert counters.captures == 2
    assert workload.presentation.prepared is prepared
    assert [(s.system_id, s.matrix_id) for s in workload.presentation.systems] == [
        (SystemId("ds-1-0"), MatrixId(1)),
        (SystemId("ds-1-1"), MatrixId(1)),
        (SystemId("ds-2-0"), MatrixId(2)),
        (SystemId("ds-2-1"), MatrixId(2)),
    ]
    assert all(s.facts == levels(2) for s in workload.presentation.systems)


def test_built_preconditioners_are_released_after_capture(
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    execute: Executor,
) -> None:
    workload = execute(
        method,
        matrix_bundles,
        prepared_method(str(method.name)),
        solver_params,
        neuralls_settings,
    )
    gc.collect()

    assert len(counters.built) == 2
    assert all(ref() is None for ref in counters.built)
    assert len(workload.presentation.systems) == 4


def test_failed_setup_is_reported_from_the_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    execute: Executor,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("setup broke")

    monkeypatch.setattr(comparison_run, "_build_scheduled", fail)
    prepared = prepared_method(str(method.name))

    workload = execute(method, matrix_bundles, prepared, solver_params, neuralls_settings)

    assert workload.presentation == ExecutedMethod(prepared, ())
    sink = _RecordingSink()
    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())
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
    execute: Executor,
) -> None:
    with pytest.raises(ValueError, match="prepared presentation"):
        execute(method, matrix_bundles, prepared_method("other"), solver_params, neuralls_settings)


@dataclass(slots=True)
class _RecordingSink:
    presentations: list[GroupPresentation] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del outcome
        self.presentations.append(presentation)


def test_group_reports_a_realized_fact_shared_by_every_matrix(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    prepared = prepared_method("jacobi")
    sink = _RecordingSink()

    report_all_levels(
        (completed_workload(prepared, {1: 2, 2: 2}),),
        sink,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    group_method = sink.presentations[0].methods[PreconditionerName("jacobi")]
    assert group_method.invariant_facts == levels(2)
    assert levels(2)[0] in group_method.compact_facts
    assert levels(2)[0] in group_method.detailed_facts


def test_group_does_not_claim_a_realized_fact_that_varies_by_matrix(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    prepared = prepared_method("jacobi")
    workload = completed_workload(prepared, {1: 2, 2: 3})
    sink = _RecordingSink()

    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    group_method = sink.presentations[0].methods[PreconditionerName("jacobi")]
    assert group_method.invariant_facts == ()
    assert all(fact.key != REALIZED_LEVELS for fact in group_method.compact_facts)
    assert [s.facts for s in workload.presentation.systems] == [levels(2), levels(3)]


def test_cohort_returns_the_prepared_presentation_of_a_known_method(
    prepared_presentation: PreparedComparisonPresentation,
) -> None:
    assert prepared_presentation.method(PreconditionerName("jacobi")).method_id == "jacobi"
    with pytest.raises(MissingMethodPresentationError):
        prepared_presentation.method(PreconditionerName("absent"))


def test_a_workload_rejects_a_presentation_of_another_method(
    prepared_method: PreparedMethodFactory,
) -> None:
    workload = completed_workload(prepared_method("jacobi"), {1: 2})

    with pytest.raises(ValueError, match="presentation belongs to method"):
        replace(workload, presentation=ExecutedMethod(prepared_method("other"), ()))


def _depth_shown(group: GroupPresentation, method: str) -> list[int]:
    return [
        int(fact.value)
        for fact in group.methods[PreconditionerName(method)].invariant_facts
        if fact.key == REALIZED_LEVELS and isinstance(fact.value, int)
    ]


def test_each_group_states_only_facts_invariant_over_its_own_systems(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    workload = completed_workload(
        prepared_method("jacobi"),
        {1: 2, 2: 3},
        recipe_by_matrix={1: ALPHA, 2: BETA},
    )
    sink = _RecordingSink()

    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    dataset = group_datasets[DatasetId("ds")]
    by_scope = [(group.scope, _depth_shown(group, "jacobi")) for group in sink.presentations]
    assert by_scope == [
        (RecipeScope(dataset, ALPHA), [2]),
        (RecipeScope(dataset, BETA), [3]),
        (DatasetScope(dataset), []),
        (AllDatasetsScope(), []),
    ]


def test_a_group_with_an_unrealized_system_states_no_realized_fact(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    workload = completed_workload(
        prepared_method("jacobi"), {1: 2, 2: 2}, realized_matrices=frozenset({2})
    )
    sink = _RecordingSink()

    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    assert all(_depth_shown(group, "jacobi") == [] for group in sink.presentations)


def test_group_lists_only_methods_with_systems_in_that_group(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    alpha = completed_workload(prepared_method("jacobi_a"), {1: 2}, recipe_by_matrix={1: ALPHA})
    beta = completed_workload(prepared_method("jacobi_b"), {2: 2}, recipe_by_matrix={2: BETA})
    sink = _RecordingSink()

    report_all_levels(
        (alpha, beta), sink, datasets=group_datasets, notation=NotationPolicy.default()
    )

    dataset = group_datasets[DatasetId("ds")]
    methods_by_scope = {group.scope: set(group.methods) for group in sink.presentations}
    assert methods_by_scope[RecipeScope(dataset, ALPHA)] == {"jacobi_a"}
    assert methods_by_scope[RecipeScope(dataset, BETA)] == {"jacobi_b"}
    assert methods_by_scope[AllDatasetsScope()] == {"jacobi_a", "jacobi_b"}


def test_partial_setup_failure_keeps_only_the_successful_matrix_systems(
    monkeypatch: pytest.MonkeyPatch,
    counters: _Counters,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    execute: Executor,
) -> None:
    succeed = comparison_run._build_scheduled

    def fail_first_matrix(
        service: PreconditionerService, matrix: torch.Tensor, cfg: PreconditionerConfig
    ) -> dict[str, Preconditioner]:
        if float(matrix[0, 0]) == 2.0:
            raise RuntimeError("setup broke on matrix 1")
        return succeed(service, matrix, cfg)

    monkeypatch.setattr(comparison_run, "_build_scheduled", fail_first_matrix)

    workload = execute(
        method, matrix_bundles, prepared_method(str(method.name)), solver_params, neuralls_settings
    )

    assert [(s.system_id, s.matrix_id) for s in workload.presentation.systems] == [
        (SystemId("ds-2-0"), MatrixId(2)),
        (SystemId("ds-2-1"), MatrixId(2)),
    ]
    assert counters.setups == 1


def test_workload_rejects_realizations_of_systems_it_did_not_plan(
    prepared_method: PreparedMethodFactory,
) -> None:
    workload = completed_workload(prepared_method("jacobi"), {1: 2})
    foreign = SystemRealization(SystemId("elsewhere"), MatrixId(9), levels(2))

    with pytest.raises(ValueError, match="did not plan"):
        replace(workload, presentation=ExecutedMethod(workload.presentation.prepared, (foreign,)))


def test_a_labelling_failure_is_loud_and_names_the_method_and_matrix(
    counters: _Counters,
    monkeypatch: pytest.MonkeyPatch,
    method: ResolvedMethod,
    matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    execute: Executor,
) -> None:
    boom = RuntimeError("cannot read hierarchy")

    def failing(precond: Preconditioner) -> tuple[ScientificFact, ...]:
        raise boom

    monkeypatch.setattr(_preconditioner_setup, "describe_realized_preconditioner", failing)

    with pytest.raises(RealizedFactCaptureError) as raised:
        execute(
            method,
            matrix_bundles,
            prepared_method(str(method.name)),
            solver_params,
            neuralls_settings,
        )

    assert raised.value.__cause__ is boom
    assert raised.value.context == ComparisonMatrixFailureContext(
        phase=ComparisonMatrixPhase.REALIZED_FACT_CAPTURE,
        method=PreconditionerName(str(method.name)),
        dataset=DatasetId("ds"),
        matrix_id=MatrixId(1),
    )
    assert str(raised.value) == (
        "Realized facts could not be read (ComparisonMatrixFailureContext("
        "phase=<ComparisonMatrixPhase.REALIZED_FACT_CAPTURE: 'realized_fact_capture'>, "
        "method='jacobi', dataset='ds', matrix_id=1)): cannot read hierarchy"
    )
