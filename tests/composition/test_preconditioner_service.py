"""`PreconditionerService` builds the implementation matching the matrix's layout."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torchalg.preconditioners.implementations import JacobiPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)

from neuralls.composition.comparison._preconditioner_setup import PreconditionerService
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from tests.fakes.fit_data import FakeFitData


@pytest.fixture
def diagonal_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD diagonal matrix."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def service() -> PreconditionerService:
    return PreconditionerService(FakeFitData())


@pytest.mark.parametrize(
    ("to_layout", "expected"),
    [
        (torch.Tensor.to_sparse_csr, SparseJacobiPreconditioner),
        (torch.Tensor.contiguous, JacobiPreconditioner),
    ],
    ids=["csr", "dense"],
)
def test_service_builds_the_implementation_of_the_matrix_layout(
    to_layout: Callable[[torch.Tensor], torch.Tensor],
    expected: type,
    diagonal_matrix: torch.Tensor,
    service: PreconditionerService,
) -> None:
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    precond = service.create_preconditioner(to_layout(diagonal_matrix), config)

    assert type(precond) is expected
