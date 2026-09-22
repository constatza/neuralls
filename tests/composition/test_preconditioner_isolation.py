"""One failing preconditioner must never abort the rest of a comparison."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison.comparison_run import _evaluate_preconditioner
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)


class _FailingService:
    """Preconditioner service whose construction raises a non-Runtime/Value error."""

    def create_preconditioner(self, matrix: torch.Tensor, cfg: object) -> object:
        raise KeyError("checkpoint key missing")


@pytest.fixture
def system_matrix() -> torch.Tensor:
    """Seeded SPD matrix (n=12)."""
    generator = torch.Generator().manual_seed(0)
    m = torch.randn(12, 12, dtype=torch.float64, generator=generator)
    return m @ m.T + 12 * torch.eye(12, dtype=torch.float64)


@pytest.fixture
def system_rhs(system_matrix: torch.Tensor) -> torch.Tensor:
    """Seeded right-hand side."""
    generator = torch.Generator().manual_seed(1)
    return system_matrix @ torch.randn(12, dtype=torch.float64, generator=generator)


@pytest.fixture
def identity_cfg() -> StandardPreconditionerConfig:
    """Config for the identity preconditioner."""
    return StandardPreconditionerConfig(type=PreconditionerType.IDENTITY, name="none")


@pytest.fixture
def solver_params() -> SolverParams:
    """Tight, cheap solver parameters."""
    return SolverParams(
        rtol=1e-8, atol=1e-14, max_iterations=50, stopping_criterion="residual_norm", m_max=5
    )


def test_unexpected_build_error_becomes_breakdown_entry(
    identity_cfg: StandardPreconditionerConfig,
    solver_params: SolverParams,
    system_matrix: torch.Tensor,
    system_rhs: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Any exception yields a breakdown entry with the error text, never a raise."""
    entry = _evaluate_preconditioner(
        identity_cfg,
        service=_FailingService(),
        matrix=system_matrix,
        rhs=system_rhs,
        matrix_path=tmp_path,
        matrix_index=0,
        params=solver_params,
    )

    assert entry.result.breakdown is True
    assert "KeyError" in (entry.result.error or "")


def test_solver_stage_error_is_isolated(
    identity_cfg: StandardPreconditionerConfig,
    solver_params: SolverParams,
    system_matrix: torch.Tensor,
    system_rhs: torch.Tensor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after construction (e.g. inside the solve) is isolated too."""

    def boom(*args: object, **kwargs: object) -> object:
        raise MemoryError("simulated")

    monkeypatch.setattr(comparison_run, "run_cg_comparison", boom)
    entry = _evaluate_preconditioner(
        identity_cfg,
        service=comparison_run.PreconditionerService(),
        matrix=system_matrix,
        rhs=system_rhs,
        matrix_path=tmp_path,
        matrix_index=0,
        params=solver_params,
    )

    assert entry.result.breakdown is True
    assert "MemoryError" in (entry.result.error or "")
