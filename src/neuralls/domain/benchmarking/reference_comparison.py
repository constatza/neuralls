"""Compare every method with the temporary observed reference selection."""

from __future__ import annotations

from neuralls.domain.benchmarking.paired import compare_paired, paired_points
from neuralls.domain.benchmarking.selection import reference_preconditioner
from neuralls.domain.solver.models.outcomes import Outcomes
from neuralls.domain.solver.models.statistics import MethodComparison
from neuralls.shared.types import PreconditionerName


def compare_to_reference(
    outcomes: Outcomes, *, seed: int
) -> dict[PreconditionerName, MethodComparison]:
    """Return failure-aware paired comparisons for every non-reference method.

    Explicit configured reference selection replaces the temporary observed
    selection in Task 7; only observed paired outcomes participate here.
    """
    reference = reference_preconditioner(outcomes)
    if reference is None:
        return {}
    return {
        name: MethodComparison(
            reference=reference,
            paired=compare_paired(paired_points(outcomes[reference], per_system), seed=seed),
        )
        for name, per_system in outcomes.items()
        if name != reference
    }
