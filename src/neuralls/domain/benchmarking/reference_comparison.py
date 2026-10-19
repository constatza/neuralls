"""Compare every method of a group against the group's reference method."""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.benchmarking.breakeven import break_even
from neuralls.domain.benchmarking.paired import compare_paired, paired_points
from neuralls.domain.benchmarking.selection import reference_preconditioner
from neuralls.domain.solver.models.outcomes import Outcomes
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.statistics import MethodComparison


def compare_to_reference(
    outcomes: Outcomes,
    projected: Mapping[str, ProjectedComparisonResult],
    *,
    seed: int,
) -> dict[str, MethodComparison]:
    """Per method, its break-even and paired comparison against the reference method.

    The reference is the method that solved the most systems. The reference
    itself and methods without a projection (never converged) are skipped;
    without a projection for the reference nothing can be compared.
    """
    reference = reference_preconditioner(outcomes)
    if reference is None or reference not in projected:
        return {}
    return {
        name: MethodComparison(
            reference=reference,
            break_even=break_even(projected[name], projected[reference]),
            paired=compare_paired(paired_points(outcomes[reference], outcomes[name]), seed=seed),
        )
        for name in outcomes
        if name != reference and name in projected
    }
