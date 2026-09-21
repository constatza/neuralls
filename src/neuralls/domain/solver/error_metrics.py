"""Reference-solution and energy-norm error metrics for CG comparisons."""

from __future__ import annotations

import torch
from torchalg.preconditioners.implementations.pod.weighting import a_row_norms

REFERENCE_PRECISION_MARGIN = 1e-4
MAX_REFINEMENT_STEPS = 5


def reference_solution(
    A: torch.Tensor, b: torch.Tensor, *, rtol: float, margin: float = REFERENCE_PRECISION_MARGIN
) -> torch.Tensor:
    """Direct solve of ``A x = b`` accurate well beyond the comparison tolerance.

    A direct solve is refined iteratively (residual correction) until
    ``||b - A x|| / ||b|| <= rtol * margin``, so the reference error is orders
    of magnitude below what the solvers under comparison are asked to reach.

    Args:
        A (torch.Tensor): System matrix, shape ``(n, n)``.
        b (torch.Tensor): Right-hand side, shape ``(n,)``.
        rtol (float): Relative tolerance requested from the compared solvers.
        margin (float): Required reference-to-``rtol`` ratio (default ``1e-4``).

    Returns:
        torch.Tensor: Reference solution ``x*`` in float64 (refinement to the
        target precision is impossible in lower precision).

    Raises:
        RuntimeError: If refinement cannot reach the target precision.
    """
    A, b = A.double(), b.double()
    target = rtol * margin * float(torch.linalg.vector_norm(b))
    x = torch.linalg.solve(A, b)
    for _ in range(MAX_REFINEMENT_STEPS):
        r = b - A @ x
        if float(torch.linalg.vector_norm(r)) <= target:
            return x
        x = x + torch.linalg.solve(A, r)
    if float(torch.linalg.vector_norm(b - A @ x)) > target:
        raise RuntimeError(
            f"Reference solution could not reach relative residual {rtol * margin:.1e}; "
            "the system is too ill-conditioned for a float64 direct solve at this rtol."
        )
    return x


def energy_error_history(
    A: torch.Tensor, x_exact: torch.Tensor, solution_vectors: torch.Tensor | None
) -> list[float] | None:
    """Relative energy-norm error per iterate, ``||e_k||_A / ||e_0||_A``.

    ``||v||_A = sqrt(v^T A v)`` is the norm CG minimizes, so this is the
    conventional convergence measure for (P)CG; normalizing by the initial
    error makes every curve start at 1.

    Args:
        A (torch.Tensor): SPD system matrix, shape ``(n, n)``.
        x_exact (torch.Tensor): Reference solution, shape ``(n,)``.
        solution_vectors (torch.Tensor | None): Traced iterates ``x_0..x_k``,
            shape ``(k + 1, n)``.

    Returns:
        list[float] | None: Relative errors, or ``None`` if iterates were not
        traced or the initial error is zero.
    """
    if solution_vectors is None:
        return None
    errors = solution_vectors.to(device=A.device, dtype=torch.float64) - x_exact.to(A.device)
    energy = a_row_norms(errors, A.double())
    if float(energy[0]) == 0:
        return None
    return (energy / energy[0]).tolist()
