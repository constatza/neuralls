"""CG solver comparison runner.

This module provides pure domain logic for running CG comparisons across multiple
preconditioners — dispatching each to the cheapest CG variant it is mathematically
compatible with (standard PCG for deterministic SPD preconditioners, Flexible CG
for iteration-varying or non-SPD ones) — plus result formatting and analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonGroupOutcome,
    ComparisonRecommendations,
    GroupAggregated,
    GroupSkippedAllCacheHit,
    ProjectedComparisonResult,
    RankedRecommendation,
    SetupSharingPolicy,
)
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL
from neuralls.shared.device import release_device_memory, track_resource_usage
from neuralls.shared.types import CostProvenance


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
            with track_resource_usage(A.device) as usage:
                x_sol, info = _solve_one(
                    A, b, x0, precond, rtol=rtol, atol=atol, maxiter=maxiter, m_max=m_max,
                    x_exact=x_exact,
                )  # fmt: skip
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
                solve_time_seconds=usage().wall_time_seconds,
                solve_peak_memory_bytes=usage().peak_memory_bytes,
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
    """Summarize best-performing CG combinations by preconditioner and overall.

    Args:
        results: CG comparison results.

    Returns:
        Typed ranked recommendations with an overall best entry.
    """
    ranked: list[RankedRecommendation] = []
    for label, info in results.items():
        if not info.converged:
            continue
        ranked.append(
            RankedRecommendation(
                label=label,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                breakdown=info.breakdown,
            )
        )

    ranked = sorted(ranked, key=lambda entry: entry.residual)
    overall_best = ranked[0] if ranked else None

    return ComparisonRecommendations(
        ranked=tuple(ranked),
        overall_best=overall_best,
    )


def aggregate_projected_results(
    results_by_system: Mapping[str, dict[str, CGComparisonResult]],
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> dict[str, ProjectedComparisonResult]:
    """Aggregate sibling systems' results into one scaling-law projection per preconditioner.

    Pure aggregation over results a comparison sweep already produced — never
    runs anything itself. Groups by preconditioner name across
    `results_by_system`'s sibling systems (each one a single system's
    `dict[str, CGComparisonResult]`, as `ComparisonOutcome.payload.results`
    already shapes it), then reduces each group with `sharing_by_name`'s
    pre-computed policy for that preconditioner — the sharing decision
    itself needs the live `torchalg.Preconditioner` object (to check
    `isinstance(precond, StaticSetupPreconditioner)`), which no longer
    exists by aggregation time, so the execution layer computes it while
    building/solving and passes it in here.

    Args:
        results_by_system: One `dict[str, CGComparisonResult]` per sibling
            system actually run (5-7 in practice, never N), keyed by
            anything stable (e.g. the system's `comparison_id`) — only the
            values are used.
        sharing_by_name: Each preconditioner name's `SetupSharingPolicy`,
            computed by the execution layer during this same sweep.

    Returns:
        One `ProjectedComparisonResult` per preconditioner name present in
        every system's results.

    Raises:
        KeyError: If a preconditioner name present in `results_by_system`
            has no entry in `sharing_by_name`.
    """
    grouped: dict[str, list[CGComparisonResult]] = {}
    for system_results in results_by_system.values():
        for name, result in system_results.items():
            grouped.setdefault(name, []).append(result)

    return {
        name: _aggregate_one_preconditioner(name, group, sharing_by_name[name])
        for name, group in grouped.items()
    }


def _aggregate_one_preconditioner(
    name: str,
    results: list[CGComparisonResult],
    sharing: SetupSharingPolicy,
) -> ProjectedComparisonResult:
    """Reduce one preconditioner's sibling results into its projection.

    A `generation_cost` with `CostProvenance.UNAVAILABLE` is excluded from
    the sum rather than counted as its (fake, near-zero) reported value —
    same treatment `CGComparisonResult.total_time_seconds` already gives an
    unavailable cost, so an unknown cost never silently reads as zero here
    either. In practice every sibling shares the same provenance (same
    config, same dataset manifest lookup), so this never actually mixes
    real and unavailable values within one preconditioner's group.

    Args:
        name: Preconditioner name shared by every result in `results`.
        results: One `CGComparisonResult` per sibling system.
        sharing: This preconditioner's pre-computed sharing policy.

    Returns:
        ProjectedComparisonResult: The aggregated projection.
    """
    real_generation_values = [
        result.generation_cost.wall_time_seconds
        for result in results
        if result.generation_cost is not None
        and result.generation_cost.provenance is not CostProvenance.UNAVAILABLE
    ]
    return ProjectedComparisonResult(
        preconditioner_name=name,
        n_systems=len(results),
        setup_sharing=sharing,
        setup_cost_total_seconds=sum(result.setup_cost for result in results),
        generation_cost_total_seconds=(
            sum(real_generation_values) if real_generation_values else None
        ),
        total_solve_time_seconds=sum(result.solve_time_seconds or 0.0 for result in results),
        total_iterations=sum(result.iterations for result in results),
    )


def classify_group_aggregation(
    group_id: str,
    n_in_group: int,
    results_by_system: Mapping[str, dict[str, CGComparisonResult]],
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> ComparisonGroupOutcome:
    """Classify one comparison group's aggregation attempt as a closed outcome.

    Pure: takes already-filtered real results (callers drop cache hits,
    which carry no `CGComparisonResult`) plus the group's total member
    count, and decides which of the two cases applies — no MLflow or
    composition-layer types involved.

    Args:
        group_id: The `ComparisonGroup` being classified.
        n_in_group: Total members in the group (including any cache hits).
        results_by_system: One `dict[str, CGComparisonResult]` per member
            that actually produced a result this run — fewer than
            `n_in_group` when some members were cache hits.
        sharing_by_name: Each preconditioner name's `SetupSharingPolicy`.

    Returns:
        `GroupAggregated` if at least one member contributed a result,
        else `GroupSkippedAllCacheHit`.
    """
    if not results_by_system:
        return GroupSkippedAllCacheHit(group_id=group_id, n_in_group=n_in_group)
    projected = aggregate_projected_results(results_by_system, sharing_by_name)
    return GroupAggregated(
        group_id=group_id,
        projected=projected,
        n_aggregated=len(results_by_system),
        n_in_group=n_in_group,
    )
