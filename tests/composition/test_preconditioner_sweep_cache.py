"""Tests for `SweepPreconditionerCache` reuse through `_run_preconditioner`.

Covers the three cases the cache must tell apart:
- A matrix-agnostic preconditioner (Identity, a `StaticSetupPreconditioner`)
  is reused across *different* matrices.
- A matrix-dependent preconditioner (Jacobi) is reused only when the matrix
  is literally the same across calls.
- A matrix-dependent preconditioner is never reused across *different*
  matrices, even with a cache present.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    SweepPreconditionerCache,
)
from neuralls.composition.comparison.comparison_run import _run_preconditioner
from neuralls.composition.comparison.models import PreconditionerComparisonEntry
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings

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
    cache: SweepPreconditionerCache,
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
    """Identity is matrix-agnostic, so it is shareable even when same_matrix=False."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    cache = SweepPreconditionerCache(same_matrix=False)
    matrix_a = _diagonal([2.0, 4.0, 8.0])
    matrix_b = _diagonal([1.0, 1.0, 9.0])
    rhs = torch.ones(3, dtype=torch.float64)

    first = _run(cfg, matrix=matrix_a, rhs=rhs, settings=neuralls_settings, cache=cache)
    second = _run(cfg, matrix=matrix_b, rhs=rhs, settings=neuralls_settings, cache=cache)

    assert cache.shared_names() == frozenset({"identity"})
    assert cache.get("identity") is not None
    assert first.result.setup_cost == second.result.setup_cost


def test_matrix_setup_preconditioner_is_reused_only_for_the_same_matrix(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Jacobi is matrix-dependent; sharing is valid only when same_matrix=True."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    cache = SweepPreconditionerCache(same_matrix=True)
    matrix = _diagonal([2.0, 4.0, 8.0])
    rhs_a = torch.ones(3, dtype=torch.float64)
    rhs_b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    first = _run(cfg, matrix=matrix, rhs=rhs_a, settings=neuralls_settings, cache=cache)
    cached_entry = cache.get("jacobi")
    assert cached_entry is not None
    first_preconditioner = cached_entry.preconditioner

    second = _run(cfg, matrix=matrix, rhs=rhs_b, settings=neuralls_settings, cache=cache)

    assert cache.shared_names() == frozenset({"jacobi"})
    # The second call reused the exact same built object — never rebuilt.
    reused_entry = cache.get("jacobi")
    assert reused_entry is not None
    assert reused_entry.preconditioner is first_preconditioner
    assert first.result.setup_cost == second.result.setup_cost


def test_matrix_setup_preconditioner_is_rebuilt_for_a_different_matrix(
    neuralls_settings: NeurallsSettings,
) -> None:
    """A matrix-varying sweep (same_matrix=False) never caches a MatrixSetupPreconditioner."""
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=AggregationCoarseningConfig(theta=0.25),
        n_levels=2,
    )
    cache = SweepPreconditionerCache(same_matrix=False)
    n = 50
    matrix_a = (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )
    matrix_b = matrix_a * 3.0
    rhs = torch.ones(n, dtype=torch.float64)

    _run(cfg, matrix=matrix_a, rhs=rhs, settings=neuralls_settings, cache=cache)
    _run(cfg, matrix=matrix_b, rhs=rhs, settings=neuralls_settings, cache=cache)

    assert cache.shared_names() == frozenset()
    assert cache.get("amg_agg") is None
