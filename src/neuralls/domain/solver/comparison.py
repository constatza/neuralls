"""CG solver comparison runner.

This module provides pure domain logic for running CG comparisons across multiple
preconditioners — dispatching each to the cheapest CG variant it is mathematically
compatible with (standard PCG for deterministic SPD preconditioners, Flexible CG
for iteration-varying or non-SPD ones) — plus result formatting and analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
import torch
from loguru import logger
from torchalg import flexible_cg, pcg
from torchalg.models.result import SolverResult
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.base import Preconditioner
from torchalg.utils.device import resolve_device

from neuralls.domain.solver.error_metrics import (
    REFERENCE_PRECISION_MARGIN,
    extract_energy_error,
    reference_solution,
    relative_exact_error,
)
from neuralls.domain.solver.metrics_table import (
    ReportLevel,
    SystemTag,
    build_metrics_frame,
    label_for_tags,
    label_key,
    report_key,
    report_level,
    system_counts,
    time_to_solution,
    time_to_solution_by_system,
)
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonGroupOutcome,
    ComparisonRecommendations,
    GroupAggregated,
    GroupSkippedAllCacheHit,
    RankedRecommendation,
    SetupSharingPolicy,
    render_group_id,
)
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL
from neuralls.shared.device import release_device_memory, repeat_timed
from neuralls.shared.timing import TimingRepeatPolicy


class CGAlgorithm(StrEnum):
    """CG variant a preconditioner is mathematically compatible with."""

    PCG = "pcg"
    FCG = "fcg"


def run_cg_comparison(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    preconditioners: Mapping[str, Preconditioner],
    x0: torch.Tensor | None = None,
    x_exact: torch.Tensor | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    maxiter: int = 100,
    m_max: int = DEFAULT_M_MAX,
    reference_precision_margin: float = REFERENCE_PRECISION_MARGIN,
    timing: TimingRepeatPolicy = TimingRepeatPolicy(),  # noqa: B008 -- frozen, so safely shared
) -> dict[str, CGComparisonResult]:
    """Run CG with multiple preconditioners for comparison.

    Each preconditioner is routed to the CG variant it is mathematically
    compatible with: standard PCG's two-term recurrence for preconditioners
    that are deterministic and exactly SPD for the whole solve
    (``Preconditioner.requires_flexible_cg`` is ``False``), or Flexible CG's
    explicit Gram-Schmidt orthogonalization for preconditioners that vary per
    iteration or are not exactly SPD (neural preconditioners, non-linear AMG,
    ``ScheduledPreconditioner``). This keeps the comparison fair by giving
    every preconditioner the correct — and cheapest — algorithm for its own
    mathematical properties, rather than forcing a single algorithm on all of
    them; every preconditioner still sees the same initial guess and system.
    The function executes exactly the supplied mapping. Selection of baselines
    or other comparison participants belongs to the caller's typed
    configuration, not to solver execution.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor.
        preconditioners: Exact mapping of result identifiers to preconditioner
            instances to execute; no entries are injected or removed.
        x0: Initial guess (defaults to zero).
        x_exact: Precomputed reference solution (any device/dtype; matched to
            ``A``'s before use), from a prior ``compute_reference_solution()``
            call — pass this when comparing multiple preconditioners against
            the same ``A``/``b`` so the dense reference solve runs once
            instead of once per call. ``None`` (default) computes it here.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        maxiter: Maximum iterations.
        m_max: FCG orthogonalization window, used only for preconditioners
            routed to ``flexible_cg``; ignored for preconditioners routed to
            ``pcg``.
        reference_precision_margin: How many orders of magnitude tighter than
            ``rtol`` the host reference solution must be. Ignored when
            ``x_exact`` is given.
        timing: How often each solve is timed. The default is one timed run.

    Returns:
        Dict mapping preconditioner names to CGComparisonResult.

    Example:
        >>> from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner
        >>> preconditioners = {
        ...     "identity": Identity(),
        ...     "jacobi": JacobiPreconditioner(A),
        ... }
        >>> results = run_cg_comparison(A, b, preconditioners=preconditioners)
    """
    device = resolve_device()
    A = A.to(device)
    b = b.to(device)
    x0 = torch.zeros_like(b, dtype=A.dtype, device=A.device) if x0 is None else x0.to(device)
    x0_base = x0.detach().clone()

    if x_exact is None:
        x_exact = compute_reference_solution(
            A.detach(), b.detach(), rtol=rtol, margin=reference_precision_margin
        )
    if x_exact is not None:
        x_exact = x_exact.to(device=A.device, dtype=A.dtype)

    results: dict[str, CGComparisonResult] = {}

    for precond_name, precond in preconditioners.items():
        try:
            timed = repeat_timed(
                lambda precond=precond: _solve_one(
                    A, b, x0, precond, rtol=rtol, atol=atol, maxiter=maxiter, m_max=m_max,
                    x_exact=x_exact,
                ),
                timing,
                device=A.device,
                fingerprint=lambda solved: solved[1].iterations,
            )  # fmt: skip
            x_sol, info = timed.value
        except (ValueError, RuntimeError) as solver_exc:
            result = CGComparisonResult(
                x=_to_numpy(x0_base),
                converged=False,
                iterations=0,
                residual=float("inf"),
                residual_abs=float("inf"),
                residual_history_rel=[],
                residual_history_abs=[],
                preconditioner=precond_name,
                initial_guess=_to_numpy(x0_base),
                exact_error=None,
                rhs_norm=float(torch.linalg.vector_norm(b)),
                breakdown=False,
                error=f"CG solver failed: {solver_exc}",
            )
        else:
            exact_error = relative_exact_error(x_sol, x_exact) if x_exact is not None else None
            error_history, error_bound = _safe_extract_energy_error(info)

            rhs_norm = info.rhs_norm
            residual: list[float] = list(info.residual_history_abs or (info.residual_abs,))
            residual_rel: list[float] = (
                [r / rhs_norm for r in residual] if rhs_norm > 0 else list(residual)
            )

            result = CGComparisonResult(
                x=_to_numpy(x_sol),
                converged=info.converged,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                residual_history_rel=residual_rel,
                residual_history_abs=residual,
                preconditioner=precond_name,
                initial_guess=_to_numpy(x0_base),
                exact_error=exact_error,
                rhs_norm=info.rhs_norm,
                breakdown=info.breakdown,
                error_history_a_rel=error_history,
                error_bound_a_rel=error_bound,
                solve_time_seconds=timed.seconds,
                solve_peak_memory_bytes=timed.peak_memory_bytes,
                solve_time_samples=timed.samples_seconds,
            )

        results[precond_name] = result
        release_device_memory()

    return results


def compute_reference_solution(
    A: torch.Tensor, b: torch.Tensor, *, rtol: float, margin: float
) -> torch.Tensor | None:
    """Jacobi-PCG reference solution, on whichever device ``A``/``b`` are on.

    A single ``A``/``b`` pair has one true solution, so callers comparing
    multiple preconditioners against the same system should call this once
    and pass the result to every ``run_cg_comparison(..., x_exact=...)`` call
    instead of letting each one recompute this Jacobi-PCG solve from scratch.

    Returns:
        The reference solution, or ``None`` (logged) if it cannot be computed —
        the comparison then runs without exact-error metrics instead of failing.
    """
    try:
        return reference_solution(A, b, rtol=rtol, margin=margin)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Reference solution unavailable; exact-error metrics skipped: {}", exc)
        return None


def _safe_extract_energy_error(
    info: SolverResult,
) -> tuple[list[float] | None, list[float] | None]:
    """Resilience wrapper around ``extract_energy_error``.

    Returns:
        ``(error_history_a_rel, error_bound_a_rel)``, or ``(None, None)`` if
        extraction fails — a metrics failure must never discard an otherwise
        valid solve.
    """
    try:
        return extract_energy_error(info)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Energy-norm error metrics skipped: {}", exc)
        return None, None


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


def format_results_summary(results: dict[str, CGComparisonResult]) -> str:
    """Format CG results into a readable summary.

    Status codes follow SciPy convention:
    - "ok": converged (info=0)
    - "fail": did not converge (info>0), either max iterations or breakdown

    Args:
        results: CG comparison results.

    Returns:
        Formatted summary string.
    """
    return "\n".join(
        ["CG comparison results:"]
        + [_format_result_line(name, result) for name, result in results.items()]
    )


def _format_result_line(name: str, result: CGComparisonResult) -> str:
    """Format one preconditioner's result as a single summary line.

    Args:
        name: Preconditioner name.
        result: Its CG comparison result.

    Returns:
        One formatted line, e.g. ``"- jacobi  status=ok  iters=5  rel_res=1.0e-08"``.
    """
    status = "ok" if result.converged else "fail"
    line = (
        f"- {name:<18} status={status:<4} iters={result.iterations:>3}"
        f"  rel_res={result.residual:.3e}"
    )
    if result.residual_abs is not None:
        line += f" (abs={result.residual_abs:.3e})"
    if result.exact_error is not None:
        line += f"  exact_err={result.exact_error:.3e}"
    if result.error:
        line += f"  note={result.error}"
    if result.breakdown:
        line += "  note=breakdown_post_convergence" if result.converged else "  note=breakdown"
    return line


def summarize_best_combinations(
    results: dict[str, CGComparisonResult],
) -> ComparisonRecommendations:
    """Rank converged CG combinations by time-to-solution.

    Every converged run stops just below the tolerance, so its final residual
    says nothing about quality; what differs is the cost of getting there.
    Entries are ordered by total time (generation + setup + solve), then by
    iterations; an entry whose time is incomparable ranks after all timed
    ones.

    Args:
        results: CG comparison results.

    Returns:
        Typed ranked recommendations with an overall best entry.
    """
    ranked = sorted(
        (
            RankedRecommendation(
                label=label,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                breakdown=info.breakdown,
                total_time_seconds=time_to_solution(info),
            )
            for label, info in results.items()
            if info.converged
        ),
        key=lambda entry: (
            entry.total_time_seconds is None,
            entry.total_time_seconds or 0.0,
            entry.iterations,
        ),
    )
    return ComparisonRecommendations(
        ranked=tuple(ranked),
        overall_best=ranked[0] if ranked else None,
    )


def classify_level(
    planned: Sequence[SystemTag],
    results_by_system: Mapping[SystemTag, Mapping[str, CGComparisonResult]],
    sharing_by_name: Mapping[str, SetupSharingPolicy],
    level: ReportLevel,
) -> tuple[ComparisonGroupOutcome, ...]:
    """Reduce one report level to a closed outcome per report group.

    Pure: ``planned`` names every system of the case, ``results_by_system``
    only those that produced results this run (cache hits and failures carry
    none). A group with at least one measured system is aggregated; one with
    none is `GroupSkippedAllCacheHit`. A group of a single system is still
    reported (without a spread); the only statistical control is the
    per-system timing repeat policy.

    Args:
        planned: Tag of every planned system, in plan order.
        results_by_system: Results of each system that produced some this run.
        sharing_by_name: Each preconditioner name's `SetupSharingPolicy`.
        level: Which columns, besides the preconditioner, key the groups.

    Returns:
        One outcome per group of ``planned`` at ``level``, in first-seen order.

    Raises:
        KeyError: If a measured preconditioner has no entry in ``sharing_by_name``.
    """
    frame = build_metrics_frame(results_by_system)
    projected_by_label = report_level(frame, level, sharing_by_name) if frame.height else {}
    times_by_label = time_to_solution_by_system(frame, level) if frame.height else {}
    counts_by_label = system_counts(frame, level) if frame.height else {}
    measured = {label_key(label): label for label in projected_by_label}

    tags_by_key: dict[tuple[str, ...], list[SystemTag]] = {}
    for tag in planned:
        tags_by_key.setdefault(report_key(level, tag), []).append(tag)

    outcomes: list[ComparisonGroupOutcome] = []
    for key, tags in tags_by_key.items():
        label = measured.get(key)
        if label is None:
            skipped_label = label_for_tags(level, tags)
            outcomes.append(
                GroupSkippedAllCacheHit(
                    group_id=render_group_id(skipped_label),
                    label=skipped_label,
                    n_in_group=len(tags),
                )
            )
            continue
        counts = counts_by_label[label]
        outcomes.append(
            GroupAggregated(
                group_id=render_group_id(label),
                label=label,
                projected=projected_by_label[label],
                n_aggregated=counts.n_systems,
                n_matrices=counts.n_matrices,
                n_in_group=len(tags),
                time_to_solution_by_system=times_by_label.get(label, {}),
            )
        )
    return tuple(outcomes)
