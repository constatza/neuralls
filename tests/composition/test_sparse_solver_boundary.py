"""Solver-boundary equivalence between dense and CSR system matrices.

`run_traced_pcg` must return the same solution for a CSR operator as for its
dense twin, and every sparse-mapped preconditioner built through the factory
must apply the same action in CSR as in dense form. Fixtures are seeded and
built in memory; nothing reads repo configs or the network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array, diags, identity, kron
from torchalg.preconditioners.base import Preconditioner

from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.solvers.torchalg_runner import _to_torch_operator, run_traced_pcg
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    IC0PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.types import MatrixFormat

GRID_SIDE = 8
"""Side of the 2D Laplacian; the system has GRID_SIDE**2 = 64 unknowns."""

RNG_SEED = 20260405
"""Seed for every random vector in this module."""

SOLVE_RTOL = 1e-12
SOLVE_ATOL = 1e-14
SOLVE_MAXITER = 500
SOLUTION_TOLERANCE = 1e-8
"""Maximum allowed absolute difference between CSR and dense results."""


@pytest.fixture
def laplacian_csr() -> csr_array:
    """Symmetric positive-definite 2D 5-point Laplacian as CSR."""
    one_d = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(GRID_SIDE, GRID_SIDE))
    eye = identity(GRID_SIDE, format="csr")
    return csr_array(kron(eye, one_d) + kron(one_d, eye))


@pytest.fixture
def rhs_vector() -> np.ndarray:
    """Seeded right-hand side of the Laplacian system."""
    rng = np.random.default_rng(RNG_SEED)
    return rng.standard_normal(GRID_SIDE**2)


def _config_for(preconditioner_type: PreconditionerType) -> ConcretePreconditionerConfig:
    """Default config for each sparse-mapped preconditioner type."""
    match preconditioner_type:
        case PreconditionerType.JACOBI | PreconditionerType.ILU:
            return StandardPreconditionerConfig(
                name=str(preconditioner_type), type=preconditioner_type
            )
        case PreconditionerType.IC0:
            return IC0PreconditionerConfig(name="ic0", type=preconditioner_type)
        case PreconditionerType.AMG:
            return AMGPreconditionerConfig(
                name="amg",
                type=preconditioner_type,
                coarsening=AggregationCoarseningConfig(),
            )
        case PreconditionerType.ADAPTIVE_SA_AMG:
            return AdaptiveSAPreconditionerConfig(name="adaptive", type=preconditioner_type)
        case PreconditionerType.BOOTSTRAP_AMG:
            return BootstrapAMGPreconditionerConfig(name="bootstrap", type=preconditioner_type)
        case _:
            raise ValueError(f"No sparse-mapped test config for {preconditioner_type}")


SPARSE_MAPPED_TYPES = [
    pytest.param(PreconditionerType.JACOBI, id="jacobi"),
    pytest.param(PreconditionerType.IC0, id="ic0"),
    pytest.param(PreconditionerType.ILU, id="ilu"),
    pytest.param(PreconditionerType.AMG, id="amg-aggregation-preset"),
    pytest.param(PreconditionerType.ADAPTIVE_SA_AMG, id="adaptive-sa-amg"),
    pytest.param(PreconditionerType.BOOTSTRAP_AMG, id="bootstrap-amg"),
]


def test_run_traced_pcg_csr_solution_matches_dense(
    laplacian_csr: csr_array, rhs_vector: np.ndarray
) -> None:
    """Unpreconditioned CSR and dense solves agree within SOLUTION_TOLERANCE."""
    x0 = np.zeros_like(rhs_vector)
    dense = laplacian_csr.toarray()

    x_csr, _ = run_traced_pcg(
        laplacian_csr, rhs_vector, x0, maxiter=SOLVE_MAXITER, rtol=SOLVE_RTOL, atol=SOLVE_ATOL
    )
    x_dense, _ = run_traced_pcg(
        dense, rhs_vector, x0, maxiter=SOLVE_MAXITER, rtol=SOLVE_RTOL, atol=SOLVE_ATOL
    )

    np.testing.assert_allclose(x_csr, x_dense, atol=SOLUTION_TOLERANCE, rtol=0.0)
    np.testing.assert_allclose(
        laplacian_csr @ x_csr, rhs_vector, atol=SOLUTION_TOLERANCE * GRID_SIDE, rtol=0.0
    )


def test_to_torch_operator_builds_csr_layout_for_csr_input(laplacian_csr: csr_array) -> None:
    """A CSR input becomes a sparse CSR tensor with the same nnz and shape."""
    operator = _to_torch_operator(laplacian_csr)

    assert operator.layout == torch.sparse_csr
    assert operator.dtype == torch.float64
    assert tuple(operator.shape) == laplacian_csr.shape
    assert operator._nnz() == laplacian_csr.nnz


def test_to_torch_operator_keeps_dense_input_dense(laplacian_csr: csr_array) -> None:
    """A dense input stays a dense float64 tensor."""
    operator = _to_torch_operator(laplacian_csr.toarray())

    assert operator.layout == torch.strided
    assert operator.dtype == torch.float64


@pytest.mark.parametrize("preconditioner_type", SPARSE_MAPPED_TYPES)
def test_sparse_preconditioner_apply_matches_dense(
    preconditioner_type: PreconditionerType,
    laplacian_csr: csr_array,
    rhs_vector: np.ndarray,
) -> None:
    """Each sparse-mapped preconditioner applies the same action in CSR and dense form."""
    config = _config_for(preconditioner_type)
    dense_operand = torch.as_tensor(laplacian_csr.toarray(), dtype=torch.float64)
    csr_operand = _to_torch_operator(laplacian_csr)

    dense_precond: Preconditioner = create_preconditioner(dense_operand, config)
    csr_precond: Preconditioner = create_preconditioner(
        csr_operand, config, matrix_format=MatrixFormat.CSR
    )
    residual = torch.as_tensor(rhs_vector, dtype=torch.float64)

    z_dense = dense_precond.apply(residual)
    z_csr = csr_precond.apply(residual)

    torch.testing.assert_close(z_csr, z_dense, atol=SOLUTION_TOLERANCE, rtol=0.0)
