"""Fixtures for domain-level solver tests."""

from __future__ import annotations

import pytest
import torch
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    """Seeded well-conditioned SPD matrix (n=20)."""
    generator = torch.Generator().manual_seed(0)
    m = torch.randn(20, 20, dtype=torch.float64, generator=generator)
    return m @ m.T + 20 * torch.eye(20, dtype=torch.float64)


@pytest.fixture
def known_solution() -> torch.Tensor:
    """Seeded ground-truth solution x*."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(20, dtype=torch.float64, generator=generator)


@pytest.fixture
def rhs(spd_matrix: torch.Tensor, known_solution: torch.Tensor) -> torch.Tensor:
    """Right-hand side generated as b = A x*."""
    return spd_matrix @ known_solution


@pytest.fixture
def preconditioners(spd_matrix: torch.Tensor) -> dict[str, object]:
    """Identity and Jacobi preconditioners for the SPD fixture."""
    return {"none": Identity(), "jacobi": JacobiPreconditioner(spd_matrix)}
