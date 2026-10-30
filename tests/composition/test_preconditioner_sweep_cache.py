"""Setup scoping of method-major execution.

Prepared state is built once per matrix and shared by its sibling RHS values, never
across matrices. Loaded checkpoint weights have their own method-owned store and are
tested with the preloading tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    matrix_key,
)
from neuralls.composition.comparison.comparison_run import execute_method_workload
from neuralls.composition.comparison.models import MatrixRef, PlannedSystem, ResolvedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId
from tests.composition.workload_support import matrix_bundle
from tests.fakes.fit_data import FakeFitData
from tests.presentation_support import PreparedMethodFactory

_SOLVER_PARAMS = SolverParams(
    rtol=1e-6,
    atol=1e-10,
    max_iterations=50,
    stopping_criterion="residual_norm",
    m_max=10,
)


def _diagonal(values: list[float]) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.float64))


def test_inline_pod_staging_reads_arrays_once_per_method() -> None:
    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=Path("pod-fit")))
    fit_data = FakeFitData()
    service = PreconditionerService(fit_data)

    service.preload(_diagonal([1.0]), config)
    service.preload(_diagonal([2.0]), config)

    assert fit_data.dense_reads == [Path("pod-fit")]


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
) -> None:
    """Static and matrix-dependent methods are rebuilt on each matrix, never across them.

    Sibling RHS of one matrix share its single build and single measured setup event.
    """
    method = ResolvedMethod(StandardPreconditionerConfig(type=preconditioner_type))

    workload = execute_method_workload(
        method,
        (matrix_bundle(1, n_rhs=2), matrix_bundle(2, n_rhs=2)),
        prepared=prepared_method(str(method.name)),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=_SOLVER_PARAMS,
        settings=neuralls_settings,
    )

    assert len(build_count) == len(workload.setup_events) == 2
    assert len(workload.solve_attempts) == 4
    assert build_count[0] != build_count[1]


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
