"""Setup scoping of method-major execution.

Prepared state is built once per matrix and shared by its sibling RHS values, never
across matrices. A POD basis is fitted once per method, before those builds. Loaded
checkpoint weights have their own method-owned store and are tested with the preloading
tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from dlkit.infrastructure.types.split import IndexSplit
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    matrix_key,
)
from neuralls.composition.comparison.comparison_run import execute_method_workload
from neuralls.composition.comparison.method_data_origin import InlineFit, NoFitData
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixRef,
    PlannedSystem,
    ResolvedMethod,
)
from neuralls.composition.generation.ports import SplitStore
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import SetupCompleted, SetupFailed
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.cost import (
    FitFolds,
    Known,
    OfflineStages,
    StageCost,
    TrainShare,
    TrainShareOfGeneration,
)
from neuralls.domain.solver.models.cost import InlineFit as InlineFitOrigin
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import CostProvenance, DatasetId, MatrixId
from tests.composition.workload_support import matrix_bundle
from tests.fakes.fit_data import FakeFitData
from tests.fakes.split_store import InMemorySplitStore
from tests.loaders_support import CPU
from tests.presentation_support import PreparedMethodFactory
from tests.solver_support import SINGLE_TIMED_RUN

_SOLVER_PARAMS = SolverParams(
    rtol=1e-6,
    atol=1e-10,
    max_iterations=50,
    stopping_criterion="residual_norm",
    m_max=10,
    timing=SINGLE_TIMED_RUN,
)


def _diagonal(values: list[float]) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.float64))


POD_FIT_DIR = Path("pod-fit")


@pytest.fixture
def pod_fit_data(monkeypatch: pytest.MonkeyPatch) -> FakeFitData:
    """Three train-fold snapshots of the two-dof workload matrices, served without a file."""
    fit_data = FakeFitData(
        rhs=np.zeros((4, 2)), solutions=np.array([[1.0, 0.5], [2.0, 1.1], [0.5, 0.2], [9.0, 9.0]])
    )
    splits = InMemorySplitStore(
        {POD_FIT_DIR: IndexSplit(train=(0, 1, 2), validation=(), test=(3,), predict=None)}
    )
    monkeypatch.setattr(comparison_run, "FS_FIT_DATA", fit_data)
    monkeypatch.setattr(comparison_run, "FS_SPLIT_STORE", splits)
    # The stored method config digests its fit dataset's files, which this test has none of.
    monkeypatch.setattr(comparison_run, "canonical_method_config", lambda cfg: {"name": cfg.name})
    return fit_data


def _run_pod(
    rank: int,
    *,
    settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    datasets: dict[DatasetId, DatasetPresentation],
    training_runs: MlflowClient,
) -> CompletedMethodWorkload:
    pod = PODCoarseningConfig(dataset_dir=POD_FIT_DIR, rank=rank)
    # A resolved identity, as preparation gives it, so nothing digests the fit dataset's files.
    method = ResolvedMethod(AMGPreconditionerConfig(coarsening=pod), identity_digest="sha256:pod")
    return execute_method_workload(
        method,
        (matrix_bundle(1), matrix_bundle(2)),
        prepared=prepared_method(str(method.name)),
        origin=InlineFit(pod),
        datasets=datasets,
        notation=NotationPolicy.default(),
        params=_SOLVER_PARAMS,
        settings=settings,
        training_runs=training_runs,
        device=CPU,
    )


def test_a_pod_method_is_fitted_once_and_its_measured_fit_is_its_offline_fit(
    pod_fit_data: FakeFitData,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    training_runs: MlflowClient,
) -> None:
    workload = _run_pod(
        1,
        settings=neuralls_settings,
        prepared_method=prepared_method,
        datasets=group_datasets,
        training_runs=training_runs,
    )

    assert pod_fit_data.dense_reads == [POD_FIT_DIR]
    assert [type(event.outcome) for event in workload.setup_events] == [SetupCompleted] * 2
    assert isinstance(workload.offline_cost.fit, Known)
    assert workload.offline_cost.fit.provenance is CostProvenance.MEASURED
    assert workload.offline_cost.fit_origin() == InlineFitOrigin()


def test_a_failed_pod_fit_is_a_setup_failure_on_every_matrix(
    pod_fit_data: FakeFitData,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    training_runs: MlflowClient,
) -> None:
    """A rank above the snapshot and dof counts cannot be fitted; the fit is attempted once."""
    workload = _run_pod(
        5,
        settings=neuralls_settings,
        prepared_method=prepared_method,
        datasets=group_datasets,
        training_runs=training_runs,
    )

    assert pod_fit_data.dense_reads == [POD_FIT_DIR]
    assert [type(event.outcome) for event in workload.setup_events] == [SetupFailed] * 2
    assert workload.solve_attempts == ()
    assert isinstance(workload.offline_cost.fit, Known)


@pytest.fixture
def build_count(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per matrix a method workload builds (the timed unit of a build)."""
    builds: list[int] = []
    real = comparison_run._build_scheduled_and_bind

    def counting(
        service: PreconditionerService,
        matrix: torch.Tensor,
        cfg: StandardPreconditionerConfig,
        staged: tuple[np.ndarray, ...],
    ):
        builds.append(int(matrix[0, 0]))
        return real(service, matrix, cfg, staged)

    monkeypatch.setattr(comparison_run, "_build_scheduled_and_bind", counting)
    return builds


@pytest.mark.parametrize(
    "preconditioner_type", [PreconditionerType.IDENTITY, PreconditionerType.JACOBI]
)
def test_a_method_is_built_once_per_matrix_whatever_its_setup_role(
    build_count: list[int],
    preconditioner_type: PreconditionerType,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    training_runs: MlflowClient,
) -> None:
    """Static and matrix-dependent methods are rebuilt on each matrix, never across them.

    Sibling RHS of one matrix share its single build and single measured setup event.
    """
    method = ResolvedMethod(StandardPreconditionerConfig(type=preconditioner_type))

    workload = execute_method_workload(
        method,
        (matrix_bundle(1, n_rhs=2), matrix_bundle(2, n_rhs=2)),
        prepared=prepared_method(str(method.name)),
        origin=NoFitData(),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=_SOLVER_PARAMS,
        settings=neuralls_settings,
        training_runs=training_runs,
        device=CPU,
    )

    assert len(build_count) == len(workload.setup_events) == 2
    assert len(workload.solve_attempts) == 4
    assert build_count[0] != build_count[1]


def test_the_offline_cost_is_resolved_once_per_method_and_shared_by_every_matrix(
    monkeypatch: pytest.MonkeyPatch,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    training_runs: MlflowClient,
) -> None:
    offline = OfflineStages(
        generation=TrainShareOfGeneration(
            Known(8.5, 1024, CostProvenance.HISTORICAL), TrainShare(1.0, FitFolds.TRAIN)
        ),
        fit=Known(90.25, None, CostProvenance.HISTORICAL),
    )
    resolved: list[MlflowClient] = []

    def resolve(
        origin: NoFitData,
        *,
        runs: MlflowClient,
        splits: SplitStore,
        measured_fit: StageCost,
        run_costs: dict[str, Known] | None,
    ) -> OfflineStages:
        resolved.append(runs)
        return offline

    monkeypatch.setattr(comparison_run, "resolve_offline_cost", resolve)
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.JACOBI))

    workload = execute_method_workload(
        method,
        (matrix_bundle(1, n_rhs=2), matrix_bundle(2, n_rhs=2)),
        prepared=prepared_method(str(method.name)),
        origin=NoFitData(),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=_SOLVER_PARAMS,
        settings=neuralls_settings,
        training_runs=training_runs,
        device=CPU,
    )

    assert resolved == [training_runs]
    assert workload.offline_cost == offline
    assert [result.offline_cost for _, result in workload.diagnostics] == [offline] * 4


def _planned(entry_id: str, rhs_source: dict[str, object]) -> PlannedSystem:
    return PlannedSystem(
        entry=ComparisonRegistryEntry(id=entry_id, matrix_dataset="ds", rhs_source=rhs_source),
        matrix=MatrixRef(matrix_id=MatrixId(0), row=0),
        rhs_row=None,
        rhs_ordinal=0,
        provenance=SharedSingleMatrix(raw_matrix_index=0),
    )


def test_matrix_key_is_the_planned_matrix_id() -> None:
    assert matrix_key(_planned("a", {"kind": "gaussian"})) == MatrixId(0)
