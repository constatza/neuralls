"""Tests for `DatasetPreconditionerCache` reuse through `_run_preconditioner`.

Covers the cases the cache must tell apart:
- A matrix-agnostic preconditioner (Identity, a `StaticSetupPreconditioner`)
  is reused across *different* matrices of a dataset.
- A matrix-dependent preconditioner (Jacobi) is reused only for the same matrix.
- A matrix-dependent preconditioner is rebuilt for a *different* matrix.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuralls.composition.comparison._preconditioner_setup import (
    DatasetPreconditionerCache,
    PreconditionerService,
    SystemPreconditionerCache,
    matrix_key,
)
from neuralls.composition.comparison.comparison_run import _run_preconditioner
from neuralls.composition.comparison.models import (
    MatrixRef,
    PlannedSystem,
    PreconditionerComparisonEntry,
)
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.types import MatrixId

_SOLVER_PARAMS = SolverParams(
    rtol=1e-6,
    atol=1e-10,
    max_iterations=50,
    stopping_criterion="residual_norm",
    m_max=10,
)


def _diagonal(values: list[float]) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.float64))


def _run(
    cfg: ConcretePreconditionerConfig,
    *,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    settings: NeurallsSettings,
    cache: SystemPreconditionerCache,
) -> PreconditionerComparisonEntry:
    return _run_preconditioner(
        cfg,
        service=PreconditionerService(),
        matrix=matrix,
        rhs=rhs,
        matrix_path=Path("/nonexistent"),
        matrix_index=0,
        params=_SOLVER_PARAMS,
        color_key=None,
        marker_key=None,
        x_exact=None,
        settings=settings,
        cache=cache,
    )


def test_static_setup_preconditioner_is_reused_across_different_matrices(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Identity is matrix-agnostic, so it is shareable even across different matrices."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    dataset_cache = DatasetPreconditionerCache()
    cache_a = dataset_cache.for_system(MatrixId(0))
    cache_b = dataset_cache.for_system(MatrixId(1))
    matrix_a = _diagonal([2.0, 4.0, 8.0])
    matrix_b = _diagonal([1.0, 1.0, 9.0])
    rhs = torch.ones(3, dtype=torch.float64)

    first = _run(cfg, matrix=matrix_a, rhs=rhs, settings=neuralls_settings, cache=cache_a)
    second = _run(cfg, matrix=matrix_b, rhs=rhs, settings=neuralls_settings, cache=cache_b)

    assert dataset_cache.shared_names() == frozenset({"identity"})
    assert cache_b.get("identity") is cache_a.get("identity")
    assert first.result.setup_cost == second.result.setup_cost


def test_matrix_setup_preconditioner_is_reused_only_for_the_same_matrix(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Jacobi is matrix-dependent; sharing is valid only on the matrix it was built for."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    dataset_cache = DatasetPreconditionerCache()
    cache = dataset_cache.for_system(MatrixId(0))
    matrix = _diagonal([2.0, 4.0, 8.0])
    rhs_a = torch.ones(3, dtype=torch.float64)
    rhs_b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    first = _run(cfg, matrix=matrix, rhs=rhs_a, settings=neuralls_settings, cache=cache)
    cached_entry = cache.get("jacobi")
    assert cached_entry is not None
    first_preconditioner = cached_entry.preconditioner

    second = _run(cfg, matrix=matrix, rhs=rhs_b, settings=neuralls_settings, cache=cache)

    assert dataset_cache.shared_names() == frozenset({"jacobi"})
    # The second call reused the exact same built object — never rebuilt.
    reused_entry = cache.get("jacobi")
    assert reused_entry is not None
    assert reused_entry.preconditioner is first_preconditioner
    assert first.result.setup_cost == second.result.setup_cost


def test_matrix_setup_preconditioner_is_rebuilt_for_a_different_matrix(
    neuralls_settings: NeurallsSettings,
) -> None:
    """A MatrixSetupPreconditioner built for one matrix is never served to another."""
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=AggregationCoarseningConfig(theta=0.25),
        n_levels=2,
    )
    dataset_cache = DatasetPreconditionerCache()
    cache_a = dataset_cache.for_system(MatrixId(0))
    cache_b = dataset_cache.for_system(MatrixId(1))
    n = 50
    matrix_a = (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )
    matrix_b = matrix_a * 3.0
    rhs = torch.ones(n, dtype=torch.float64)

    _run(cfg, matrix=matrix_a, rhs=rhs, settings=neuralls_settings, cache=cache_a)
    _run(cfg, matrix=matrix_b, rhs=rhs, settings=neuralls_settings, cache=cache_b)

    built_for_a = cache_a.get(cfg.name)
    built_for_b = cache_b.get(cfg.name)
    assert built_for_a is not None
    assert built_for_b is not None
    assert built_for_a.preconditioner is not built_for_b.preconditioner
    assert dataset_cache.shared_names() == frozenset()


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
