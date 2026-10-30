"""One failing preconditioner must never abort the rest of a method workload."""

from __future__ import annotations

import pytest

from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison._preconditioner_setup import PreconditionerService
from neuralls.composition.comparison.models import MatrixWorkload, ResolvedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import SetupCompleted, SetupFailed, SolveFailed
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId
from tests.composition.workload_support import DIAGONAL, matrix_bundle
from tests.presentation_support import PreparedMethodFactory

FAILING_MATRIX_ID = 1
HEALTHY_MATRIX_ID = 2
BUILD_ERROR = KeyError("checkpoint key missing")


@pytest.fixture
def method() -> ResolvedMethod:
    """The identity method every workload runs."""
    return ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))


@pytest.fixture
def two_matrix_bundles() -> tuple[MatrixWorkload, MatrixWorkload]:
    """Matrix 1 then matrix 2, one RHS each."""
    return matrix_bundle(FAILING_MATRIX_ID), matrix_bundle(HEALTHY_MATRIX_ID)


@pytest.fixture
def solver_params() -> SolverParams:
    """Tight, cheap solver parameters."""
    return SolverParams(
        rtol=1e-8, atol=1e-14, max_iterations=50, stopping_criterion="residual_norm", m_max=5
    )


@pytest.fixture
def fail_build_on_first_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the service raise a non-Runtime/Value error while building for the first matrix."""
    real = PreconditionerService.create_preconditioner

    def create(self: PreconditionerService, matrix, config):
        if float(matrix[0, 0]) == DIAGONAL[0] * FAILING_MATRIX_ID:
            raise BUILD_ERROR
        return real(self, matrix, config)

    monkeypatch.setattr(PreconditionerService, "create_preconditioner", create)


def test_unexpected_build_error_is_a_setup_failure_and_later_matrices_still_run(
    fail_build_on_first_matrix: None,
    method: ResolvedMethod,
    two_matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    workload = comparison_run.execute_method_workload(
        method,
        two_matrix_bundles,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    failed, completed = workload.setup_events
    assert isinstance(failed.outcome, SetupFailed)
    assert isinstance(completed.outcome, SetupCompleted)
    assert [attempt.matrix_id for attempt in workload.solve_attempts] == [
        two_matrix_bundles[1].matrix_id
    ]
    breakdowns = {system_id: result.breakdown for system_id, result in workload.diagnostics}
    assert breakdowns == {
        two_matrix_bundles[0].members[0].tag.system_id: True,
        two_matrix_bundles[1].members[0].tag.system_id: False,
    }


def test_solver_stage_error_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
    method: ResolvedMethod,
    two_matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    """A failure inside the solve is recorded as a failed attempt, not raised."""
    from neuralls.domain.solver import comparison as solver_comparison

    def boom(*args: object, **kwargs: object) -> object:
        raise MemoryError("simulated")

    monkeypatch.setattr(solver_comparison, "_solve_one", boom)
    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *a, **k: None)

    workload = comparison_run.execute_method_workload(
        method,
        two_matrix_bundles,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert len(workload.solve_attempts) == len(two_matrix_bundles)
    assert all(isinstance(attempt.outcome, SolveFailed) for attempt in workload.solve_attempts)
