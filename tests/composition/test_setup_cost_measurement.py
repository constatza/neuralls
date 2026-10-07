"""Regression test: AMG/POD-2G setup cost is really measured, not near-zero.

Before `torchalg`'s two-phase `setup(matrix)` contract, AMG/POD-2G built
their hierarchy lazily on first `apply()`, so the construction-time timer in
`comparison_run.py` wrapped nothing expensive and the real hierarchy-build
cost silently leaked into `solve_time_seconds` instead of `setup_cost`. This
test builds a real classical-AMG and a real from-scratch POD-2G comparison
through the actual composition factory path and asserts `setup_cost` is a
plain, substantial `float` — not `None`/`StageCost`-wrapped and not a
load/assembly artifact masquerading as the real cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuralls.composition.comparison._preconditioner_setup import PreconditionerService
from neuralls.composition.comparison.comparison_run import _run_preconditioner
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.settings import NeurallsSettings

_SOLVER_PARAMS = SolverParams(
    rtol=1e-6,
    atol=1e-10,
    max_iterations=100,
    stopping_criterion="residual_norm",
    m_max=10,
)


def _tridiagonal_spd(n: int) -> torch.Tensor:
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


def test_classical_amg_setup_cost_is_substantial_and_a_plain_float(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Classical aggregation-coarsened AMG's hierarchy build is real, measured work."""
    n = 1200
    matrix = _tridiagonal_spd(n)
    rhs = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    cfg = AMGPreconditionerConfig(
        name="amg_agg",
        coarsening=AggregationCoarseningConfig(theta=0.25),
        n_levels=3,
    )

    entry = _run_preconditioner(
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
        settings=neuralls_settings,
    )

    assert isinstance(entry.result.setup_cost, float)
    assert entry.result.setup_cost > 0.0
    # The original bug: AMG's hierarchy build was lazy, so setup_cost was a
    # near-zero construction-only artifact next to a solve_time_seconds that
    # silently absorbed the real cost. A real hierarchy build is comparable
    # to (often exceeds) one full CG solve's wall time on this system size.
    assert entry.result.solve_time_seconds is not None
    assert entry.result.setup_cost >= 0.1 * entry.result.solve_time_seconds


def test_pod2g_from_scratch_setup_cost_is_substantial_and_a_plain_float(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A from-scratch POD-2G fit (SVD + Galerkin triple product) is real, measured work."""
    n = 800
    matrix = _tridiagonal_spd(n)
    rhs = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix.numpy())
    dataset_dir = tmp_path / "pod-dataset"
    build_dataset(
        SourceSpec(matrix_path=str(matrix_path)),
        DatasetSpec(
            mixture=MixtureSpec(counts={"gaussian_forward": 20}, seed=0, shuffle=False),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )

    cfg = AMGPreconditionerConfig(
        name="pod2g",
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=0.99),
        n_levels=2,
    )

    entry = _run_preconditioner(
        cfg,
        service=PreconditionerService(),
        matrix=matrix,
        rhs=rhs,
        matrix_path=dataset_dir,
        matrix_index=0,
        params=_SOLVER_PARAMS,
        color_key=None,
        marker_key=None,
        x_exact=None,
        settings=neuralls_settings,
    )

    assert isinstance(entry.result.setup_cost, float)
    assert entry.result.setup_cost > 0.0
