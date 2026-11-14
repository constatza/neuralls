"""CG comparison runner: solve each preconditioner with the cheapest compatible CG variant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import numpy as np
import torch
from torchalg import flexible_cg, pcg
from torchalg.models.result import SolverResult
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.base import Preconditioner

from neuralls.domain.solver.error_metrics import (
    ReferenceSolver,
    extract_energy_error,
    reference_solver,
    relative_exact_error,
)
from neuralls.domain.solver.models.benchmark_events import (
    ExactReference,
    ExactSolutionSolved,
    ExactSolutionUnavailable,
    MeasuredResidual,
    UnmeasuredResidual,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.energy_error import (
    EnergyError,
    EnergyErrorFailed,
    EnergyErrorUnavailable,
)
from neuralls.domain.solver.true_residual import NonFiniteResidual, measure_residual
from neuralls.shared.device import (
    FailedTimedRuns,
    RepeatAgreement,
    Unchecked,
    repeat_timed_outcome,
)


class CGAlgorithm(StrEnum):
    """CG variant a preconditioner is mathematically compatible with."""

    PCG = "pcg"
    FCG = "fcg"


def solve_system(
    A: torch.Tensor,
    b: torch.Tensor,
    preconditioner: Preconditioner,
    *,
    name: str,
    reference: Reference,
    params: SolverParams,
) -> CGComparisonResult:
    """Solve one system with one preconditioner, timed, and measure the result.

    The preconditioner is routed to the CG variant it is mathematically compatible with:
    standard PCG's two-term recurrence when it is deterministic and exactly SPD for the
    whole solve (``Preconditioner.requires_flexible_cg`` is ``False``), or Flexible CG's
    explicit Gram-Schmidt orthogonalization when it varies per iteration or is not exactly
    SPD (neural preconditioners, non-linear AMG, ``ScheduledPreconditioner``). Every method
    gets the correct and cheapest algorithm for its own properties, from the same zero
    initial guess.

    The solve runs on ``A``'s device: the caller places ``A``, ``b`` and the preconditioner.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor, on ``A``'s device.
        preconditioner: The method's built preconditioner.
        name: Result identifier recorded on the result.
        reference: The reference solution of this system, or why there is none; from
            ``compute_reference_solution``.
        params: Tolerances, iteration cap, FCG window and timing policy.

    Returns:
        The solve's result; a failed or non-finite solve is a result with ``error`` set.
    """
    x0 = torch.zeros_like(b, dtype=A.dtype, device=A.device)
    x0_base = x0.clone()  # the solvers may update x0 in place
    solve = partial(
        _solve_one, A, b, x0, preconditioner,
        rtol=params.rtol, atol=params.atol, maxiter=params.max_iterations, m_max=params.m_max,
    )  # fmt: skip
    timed = repeat_timed_outcome(
        lambda: solve(x_exact=None),
        params.timing,
        device=A.device,
        fingerprint=lambda solved: solved[1].iterations,
    )
    exact_reference = _outcome_of(reference)
    if isinstance(timed, FailedTimedRuns):
        return _unsolved_result(
            name,
            x0_base,
            b,
            exact_reference,
            error=f"CG solver failed: {timed.error}",
            seconds=timed.seconds,
            peak_memory_bytes=timed.peak_memory_bytes,
            samples_seconds=timed.samples_seconds,
            timing_agreement=Unchecked(),
        )
    x_sol, info = timed.value
    match measure_residual(A, b, x_sol):
        case NonFiniteResidual():
            return _unsolved_result(
                name,
                x0_base,
                b,
                exact_reference,
                error=f"non-finite iterate after {info.iterations} iterations",
                seconds=timed.seconds,
                peak_memory_bytes=timed.peak_memory_bytes,
                samples_seconds=timed.samples_seconds,
                timing_agreement=timed.agreement,
                breakdown=info.breakdown,
            )
        case MeasuredResidual() | UnmeasuredResidual() as true_residual:
            pass
    match reference:
        case ReferenceSolution(x=x_exact):
            x_ref = x_exact.to(device=A.device, dtype=A.dtype)
            exact_error: float | None = relative_exact_error(x_sol, x_ref)
            energy_error = _untimed_energy_error(lambda: solve(x_exact=x_ref))
        case ExactSolutionUnavailable():
            exact_error = None
            energy_error = _guarded_energy_error(lambda: info)

    rhs_norm = info.rhs_norm
    residual: list[float] = list(info.residual_history_abs or (info.residual_abs,))
    residual_rel = [r / rhs_norm for r in residual] if rhs_norm > 0 else list(residual)
    return CGComparisonResult(
        x=_to_numpy(x_sol),
        converged=info.converged,
        iterations=info.iterations,
        residual=info.residual,
        residual_abs=info.residual_abs,
        residual_history_rel=residual_rel,
        residual_history_abs=residual,
        preconditioner=name,
        initial_guess=_to_numpy(x0_base),
        exact_error=exact_error,
        rhs_norm=info.rhs_norm,
        breakdown=info.breakdown,
        energy_error=energy_error,
        true_residual=true_residual,
        exact_reference=exact_reference,
        solve_time_seconds=timed.seconds,
        solve_peak_memory_bytes=timed.peak_memory_bytes,
        solve_time_samples=timed.samples_seconds,
        timing_agreement=timed.agreement,
    )


type PreparedReference = ReferenceSolver | ExactSolutionUnavailable
"""A matrix's reference factorization, or why it could not be built."""


@dataclass(frozen=True, slots=True)
class ReferenceSolution:
    """A right-hand side's reference solution ``x*`` and the residual it achieved."""

    x: torch.Tensor
    outcome: ExactSolutionSolved


type Reference = ReferenceSolution | ExactSolutionUnavailable
"""One system's reference solution, or why there is none."""


def _outcome_of(reference: Reference) -> ExactReference:
    """The stored outcome of a reference, without the solution vector."""
    match reference:
        case ReferenceSolution(outcome=outcome):
            return outcome
        case ExactSolutionUnavailable():
            return reference


def prepare_reference(A: torch.Tensor) -> PreparedReference:
    """Factor ``A`` once for the reference solves of all its right-hand sides.

    A failed factorization is a domain outcome, not an error: every system of the matrix
    then runs without exact-error metrics, and the reason is returned for the caller.
    """
    try:
        return reference_solver(A)
    except (RuntimeError, ValueError) as exc:
        return ExactSolutionUnavailable(f"reference factorization failed: {exc}")


def compute_reference_solution(
    reference: PreparedReference, b: torch.Tensor, *, target_rel: float
) -> Reference:
    """Reference solution for one right-hand side, on the factored matrix's device.

    One ``A``/``b`` pair has one true solution, so the caller computes it once per system;
    callers solving several right-hand sides of one matrix share one ``prepare_reference``
    result between them.

    Args:
        reference: The matrix's factorization, or why it failed.
        b: Right-hand side.
        target_rel: Relative residual the reference must reach
            (``reference_target(rtol, margin).applied_rel``).

    Returns:
        The solution with its achieved true residual, or why it could not be computed; the
        comparison then runs without exact-error metrics instead of failing.
    """
    if isinstance(reference, ExactSolutionUnavailable):
        return reference
    try:
        x = reference.solve(b, target_rel=target_rel)
    except (RuntimeError, ValueError) as exc:
        return ExactSolutionUnavailable(f"reference solve failed: {exc}")
    match measure_residual(reference.matrix, b.double().to(reference.matrix.device), x):
        case MeasuredResidual(true_residual_rel=residual):
            return ReferenceSolution(x, ExactSolutionSolved(residual))
        case UnmeasuredResidual():
            return ExactSolutionUnavailable("zero right-hand side")
        case NonFiniteResidual():
            return ExactSolutionUnavailable("reference solution is not finite")


def _unsolved_result(
    precond_name: str,
    x0: torch.Tensor,
    b: torch.Tensor,
    exact_reference: ExactReference,
    *,
    error: str,
    seconds: float,
    peak_memory_bytes: int,
    samples_seconds: tuple[float, ...],
    timing_agreement: RepeatAgreement,
    breakdown: bool = False,
) -> CGComparisonResult:
    """A solve that produced no usable iterate: the initial guess, 0 iterations, no residual.

    Stored as a failed solve with an unmeasured residual, whose iteration count must be 0, so
    anything worth knowing about the attempt (the iterations a diverged solve ran) belongs in
    ``error``.
    """
    return CGComparisonResult(
        x=_to_numpy(x0),
        converged=False,
        iterations=0,
        residual=float("inf"),
        residual_abs=float("inf"),
        residual_history_rel=[],
        residual_history_abs=[],
        preconditioner=precond_name,
        initial_guess=_to_numpy(x0),
        exact_error=None,
        rhs_norm=float(torch.linalg.vector_norm(b)),
        breakdown=breakdown,
        error=error,
        energy_error=EnergyErrorUnavailable("the solve produced no usable iterate"),
        true_residual=UnmeasuredResidual(),
        exact_reference=exact_reference,
        solve_time_seconds=seconds,
        solve_peak_memory_bytes=peak_memory_bytes,
        solve_time_samples=samples_seconds,
        timing_agreement=timing_agreement,
    )


def _guarded_energy_error(solved: Callable[[], SolverResult]) -> EnergyError:
    """The energy error of a solve; a metrics failure never discards a valid solve."""
    try:
        return extract_energy_error(solved())
    except (RuntimeError, ValueError) as exc:
        return EnergyErrorFailed(str(exc))


def _untimed_energy_error(
    solve_tracking_reference: Callable[[], tuple[torch.Tensor, SolverResult]],
) -> EnergyError:
    """Energy-norm error history from a second, untimed solve that tracks ``x_exact``.

    ``torchalg`` computes the exact A-norm error at every iteration when given
    ``x_exact``, which would be billed to the timed solve. The solve is
    deterministic, so repeating it here yields the same iterates without
    touching any timer.
    """
    return _guarded_energy_error(lambda: solve_tracking_reference()[1])


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    """Convert solver tensors at the reporting DTO boundary."""
    return value.detach().cpu().numpy()


def _cg_algorithm_for(preconditioner: Preconditioner) -> CGAlgorithm:
    """Classify which CG variant a preconditioner is compatible with.

    Delegates to the preconditioner's own knowledge of solver compatibility,
    following OCP: adding a new preconditioner type never requires updating
    this function.

    Args:
        preconditioner: Preconditioner instance to classify.

    Returns:
        ``CGAlgorithm.FCG`` if flexible CG is required, else ``CGAlgorithm.PCG``.
    """
    return CGAlgorithm.FCG if preconditioner.requires_flexible_cg else CGAlgorithm.PCG


def _solve_one(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    preconditioner: Preconditioner,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
    m_max: int,
    x_exact: torch.Tensor | None,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve with the CG variant this preconditioner's category requires.

    Always traces at ``TraceMode.MINIMAL`` (scalar residual history only, no
    per-iterate vectors). When ``x_exact`` is supplied, ``torchalg`` tracks the
    exact energy-norm error every iteration from it directly — no vector storage,
    no separate FULL-trace pass needed for that metric.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor.
        x0: Initial guess tensor.
        preconditioner: Preconditioner instance to solve with.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        maxiter: Maximum iterations.
        m_max: FCG orthogonalization window; unused on the PCG branch — PCG's
            own (unrelated, off-by-default) periodic-reorthogonalization
            ``m_max`` is intentionally left disabled here.
        x_exact: Reference solution (already on ``A``'s device/dtype), or
            ``None`` if unavailable.

    Returns:
        Tuple of the solved vector and solver diagnostics.
    """
    match _cg_algorithm_for(preconditioner):
        case CGAlgorithm.PCG:
            return pcg(
                A,
                b,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                preconditioner=preconditioner,
                trace_mode=TraceMode.MINIMAL,
                x_exact=x_exact,
            )
        case CGAlgorithm.FCG:
            return flexible_cg(
                A,
                b,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                preconditioner=preconditioner,
                m_max=m_max,
                trace_mode=TraceMode.MINIMAL,
                x_exact=x_exact,
            )
