"""Snapshot trajectories shared by the CG-error and Jacobi-probe strategies.

The two POD snapshot sources differ only in how a probe ``z`` is propagated:

- CG errors: ``b = A z``, ``x_0 = 0``, exactly ``window.stop`` CG iterations,
  rows ``(r_k, e_k)`` with ``e_k = z - x_k`` and ``r_k = b - A x_k`` as the solver
  returns it (``r_k = A e_k`` in exact arithmetic).
- Jacobi probes: ``e_k = (I - omega D^-1 A)^k z`` (the weighted-Jacobi error
  propagation on ``A y = 0``), rows ``(A e_k, e_k)``.

Both keep the steps a ``StepWindow`` selects, refuse a trajectory shorter than the
window, and may scale each kept row by a power of its error norm
(``scale_rows``). Every function is pure: the CG solver is injected.
"""

from __future__ import annotations

from typing import Literal, assert_never

import numpy as np
import torch
from scipy.sparse import csr_array
from torchalg.preconditioners.implementations.pod import apply_jacobi_damping_trajectory
from torchalg.sparse.preconditioners.amg.smoothers import JacobiSmoother as SparseJacobiSmoother

from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples
from neuralls.shared.types import SystemMatrix

from .counts import _build_trace_indices
from .interfaces import TracingSolverCallable
from .step_window import StepWindow
from .strategy_configs import (
    NormSnapshotScaling,
    NoSnapshotScaling,
    ProbeDistribution,
    SnapshotScaling,
)
from .trace_utils import require_full_trajectory
from .transforms import ComputeRhsTransform

TRAJECTORY_UNREACHABLE_TOLERANCE = 1e-20
"""Below float64 machine epsilon (~2.2e-16), so CG can never satisfy it and every
trajectory runs exactly `window.stop` iterations. Passed explicitly because
torchalg's own defaults (1e-6 / 1e-14) are reachable and would stop early,
producing a trajectory shorter than the window and a TrajectoryShortfallError."""


def draw_probes(
    rng: np.random.Generator, count: int, n: int, distribution: ProbeDistribution
) -> np.ndarray:
    """Draw ``count`` probe vectors of length ``n``, not normalized.

    Args:
        rng: Seeded random generator; the draw consumes its stream.
        count: Number of probe vectors.
        n: Vector length (the system dimension).
        distribution: ``"gaussian"`` (standard normal entries) or ``"rademacher"``
            (uniform +-1 entries).

    Returns:
        Probes, shape (count, n), dtype float64.
    """
    match distribution:
        case "gaussian":
            return rng.standard_normal((count, n))
        case "rademacher":
            return rng.choice([-1.0, 1.0], size=(count, n))
        case _:
            assert_never(distribution)


def _tensor_trace_to_numpy(value: torch.Tensor | None) -> np.ndarray:
    if value is None:
        return np.array([])
    return value.detach().cpu().numpy()


def cg_error_traces(
    matrix: SystemMatrix,
    rhs: np.ndarray,
    solutions: np.ndarray,
    window: StepWindow,
    solver: TracingSolverCallable,
) -> ErrorTraceSamples:
    """Run CG from ``x_0 = 0`` on every ``(b, x*)`` pair and keep the window's steps.

    Args:
        matrix: System matrix A.
        rhs: Right-hand sides b, shape (systems, n).
        solutions: Known solutions x*, shape (systems, n); the errors are ``x* - x_k``.
        window: The steps to run (``stop``) and keep.
        solver: Tracing CG solver returning residual and iterate histories.

    Returns:
        The kept ``(r_k, x_k, e_k)`` rows of every system, in system order.

    Raises:
        TrajectoryShortfallError: If a system's trajectory ends before ``window.stop``.
    """
    n = matrix.shape[0]
    expected_indices = window.resolve_indices(window.stop + 1)
    residual_blocks: list[np.ndarray] = []
    iterate_blocks: list[np.ndarray] = []
    error_blocks: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    iteration_indices: list[np.ndarray] = []

    for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs, solutions)):
        _, info = solver(
            matrix,
            rhs_vec,
            np.zeros(n, dtype=np.float64),
            maxiter=window.stop,
            rtol=TRAJECTORY_UNREACHABLE_TOLERANCE,
            atol=TRAJECTORY_UNREACHABLE_TOLERANCE,
        )
        residual_seq, indices = window.select_with_indices(
            _tensor_trace_to_numpy(info.residual_vectors)
        )
        require_full_trajectory(indices, expected_indices, sample_idx, window.stop)
        iterates_full = _tensor_trace_to_numpy(info.solution_vectors)
        iterate_seq = window.select(iterates_full) if iterates_full.size > 0 else iterates_full

        residual_blocks.append(residual_seq)
        iterate_blocks.append(iterate_seq)
        error_blocks.append(np.array([true_sol - x_k for x_k in iterate_seq], dtype=np.float64))
        sidx, iidx = _build_trace_indices(sample_idx, indices)
        sample_indices.append(sidx)
        iteration_indices.append(iidx)

    return ErrorTraceSamples(
        residuals=np.vstack(residual_blocks),
        solutions_current=np.vstack(iterate_blocks),
        errors=np.vstack(error_blocks),
        true_solutions=solutions,
        sample_indices=np.concatenate(sample_indices),
        iteration_indices=np.concatenate(iteration_indices),
    )


def _sparse_damping_trajectories(
    matrix: csr_array, probes: torch.Tensor, *, omega: float | None, steps: int
) -> torch.Tensor:
    """Weighted-Jacobi trajectories for a CSR matrix, without densifying it.

    Mirrors ``apply_jacobi_damping_trajectory`` (dense torch only) with the
    torchalg sparse ``JacobiSmoother``, which accepts a torch CSR operator.
    The operator is built from the scipy buffers directly, so the O(n^2)
    dense copy never exists.

    Args:
        matrix: Square SPD system matrix in CSR format.
        probes: Batch of vectors to damp, shape (n_vectors, n_dofs).
        omega: Jacobi damping factor; ``None`` selects torchalg's spectral rule.
        steps: Number of damping sweeps per vector.

    Returns:
        torch.Tensor: Trajectories, shape (n_vectors, steps + 1, n_dofs).
    """
    operator = torch.sparse_csr_tensor(
        torch.as_tensor(matrix.indptr, dtype=torch.int64),
        torch.as_tensor(matrix.indices, dtype=torch.int64),
        torch.as_tensor(matrix.data),
        size=matrix.shape,
        check_invariants=True,
    )
    smoother = SparseJacobiSmoother(omega=omega).prepare(operator)
    trajectories = []
    for row in probes:
        zero_rhs = torch.zeros_like(row)
        x = row
        history = [x]
        for _ in range(steps):
            x = smoother.smooth(zero_rhs, x, steps=1)
            history.append(x)
        trajectories.append(torch.stack(history))
    return torch.stack(trajectories)


def _damping_trajectories(
    matrix: SystemMatrix, probes: np.ndarray, *, omega: float | None, steps: int
) -> torch.Tensor:
    """Dispatch the Jacobi damping sweep on the matrix format.

    Dense input uses the torchalg dense helper; CSR input uses the sparse smoother,
    so the matrix is never densified.

    Args:
        matrix: System matrix, dense ndarray or scipy csr_array.
        probes: Probe vectors, shape (n_vectors, n_dofs).
        omega: Jacobi damping factor; ``None`` selects the spectral rule.
        steps: Number of damping sweeps per vector.

    Returns:
        torch.Tensor: Trajectories, shape (n_vectors, steps + 1, n_dofs).
    """
    probes_t = torch.as_tensor(probes)
    match matrix:
        case np.ndarray():
            return apply_jacobi_damping_trajectory(
                probes_t, torch.as_tensor(matrix), omega=omega, steps=steps
            )
        case csr_array():
            return _sparse_damping_trajectories(matrix, probes_t, omega=omega, steps=steps)


def jacobi_probe_traces(
    matrix: SystemMatrix,
    probes: np.ndarray,
    window: StepWindow,
    omega: float | None,
) -> ResidualTraceSamples:
    """Damp every probe by ``window.stop`` weighted-Jacobi sweeps and keep the window's steps.

    Args:
        matrix: System matrix A.
        probes: Initial errors z, shape (probes, n), in the matrix dtype.
        window: The sweeps to run (``stop``) and keep.
        omega: Jacobi damping factor; ``None`` selects torchalg's spectral rule.

    Returns:
        The kept ``(A e_k, e_k)`` rows of every probe, in probe order.

    Raises:
        TrajectoryShortfallError: If a probe's trajectory ends before ``window.stop``.
    """
    trajectories = _damping_trajectories(matrix, probes, omega=omega, steps=window.stop)
    rhs_transform = ComputeRhsTransform(matrix)
    expected_indices = window.resolve_indices(window.stop + 1)
    error_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    iteration_indices: list[np.ndarray] = []

    for sample_idx, trajectory in enumerate(trajectories):
        selected, indices = window.select_with_indices(trajectory.numpy())
        require_full_trajectory(indices, expected_indices, sample_idx, window.stop)
        error_blocks.append(selected)
        residual_blocks.append(rhs_transform.transform(selected))
        sidx, iidx = _build_trace_indices(sample_idx, indices)
        sample_indices.append(sidx)
        iteration_indices.append(iidx)

    return ResidualTraceSamples(
        residuals=np.vstack(residual_blocks),
        solutions=np.vstack(error_blocks),
        sample_indices=np.concatenate(sample_indices),
        iteration_indices=np.concatenate(iteration_indices),
    )


def _error_norms(
    residuals: np.ndarray, errors: np.ndarray, norm: Literal["l2", "energy"]
) -> np.ndarray:
    """Per-row ``||e_k||_2`` or ``||e_k||_A = sqrt(e_k . r_k)``, using ``r_k = A e_k``.

    Raises:
        ValueError: If an energy norm is requested and some ``e_k . r_k`` is negative
            (A not positive definite).
    """
    match norm:
        case "l2":
            return np.linalg.norm(errors, axis=1)
        case "energy":
            squared = np.einsum("ij,ij->i", errors, residuals)
            if np.any(squared < 0.0):
                raise ValueError(
                    "energy snapshot scaling needs e_k . A e_k >= 0 for every row; "
                    "the matrix is not positive definite"
                )
            return np.sqrt(squared)
        case _:
            assert_never(norm)


def scale_rows(
    residuals: np.ndarray, errors: np.ndarray, scaling: SnapshotScaling
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each row pair by ``c_k = ||e_k||^-power``, keeping ``A e_k = r_k``.

    Both ``r_k`` and ``e_k`` are multiplied by the same ``c_k``, so a training pair stays
    consistent with the matrix that generated it. A zero row has no norm to divide by
    and is left unchanged.

    Args:
        residuals: Rows ``r_k = A e_k``, shape (rows, n).
        errors: Rows ``e_k``, shape (rows, n).
        scaling: The row scaling to apply.

    Returns:
        The scaled ``(residuals, errors)``; the inputs themselves for ``NoSnapshotScaling``.
    """
    match scaling:
        case NoSnapshotScaling():
            return residuals, errors
        case NormSnapshotScaling(norm=norm, power=power):
            norms = _error_norms(residuals, errors, norm)
            nonzero = norms > 0.0
            scales = np.ones_like(norms)
            scales[nonzero] = norms[nonzero] ** -power
            return residuals * scales[:, None], errors * scales[:, None]
        case _:
            assert_never(scaling)
