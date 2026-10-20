"""Compare methods against an explicitly configured reference."""

from __future__ import annotations

from neuralls.domain.benchmarking.paired import compare_paired, paired_points
from neuralls.domain.benchmarking.selection import ConfiguredReference, ReferenceSelection
from neuralls.domain.solver.models.outcomes import Outcomes
from neuralls.domain.solver.models.statistics import MethodComparison
from neuralls.shared.types import PreconditionerName


def compare_to_reference(
    outcomes: Outcomes, *, selection: ReferenceSelection, seed: int
) -> dict[PreconditionerName, MethodComparison]:
    """Return failure-aware paired comparisons for every non-reference method.

    The selection is configuration/runtime state, never inferred from outcomes.
    """
    if not isinstance(selection, ConfiguredReference):
        return {}
    reference = selection.name
    if reference not in outcomes:
        return {}
    return {
        name: MethodComparison(
            reference=reference,
            paired=compare_paired(paired_points(outcomes[reference], per_system), seed=seed),
        )
        for name, per_system in outcomes.items()
        if name != reference
    }
