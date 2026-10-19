"""Which method is the reference and which systems all methods solved."""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.benchmarking.costs import seconds
from neuralls.domain.solver.models.outcomes import Outcomes, Solved, SystemOutcome


def reference_preconditioner(outcomes: Outcomes) -> str | None:
    """The method that solved the most systems, or `None` for an empty group.

    Ties go to the lowest summed recurring time over the method's solved
    systems with a comparable time, then to plan order. A configuration-free
    baseline for paired comparisons: the most robust method, so the systems it
    fails on are the informative ones.
    """
    ranked = {
        name: (-_n_solved(per_system), _total_seconds(per_system))
        for name, per_system in outcomes.items()
    }
    return min(ranked, key=ranked.__getitem__, default=None)


def _n_solved(per_system: Mapping[str, SystemOutcome]) -> int:
    return sum(isinstance(o, Solved) for o in per_system.values())


def _total_seconds(per_system: Mapping[str, SystemOutcome]) -> float:
    times = (seconds(o) for o in per_system.values() if isinstance(o, Solved))
    return sum(t for t in times if t is not None)


def systems_solved_by_all(outcomes: Outcomes) -> frozenset[str]:
    """Systems on which every method converged.

    Conditional means over a method's own converged systems favour methods that
    only solve easy systems (survivorship bias); comparing on this common
    subset removes it, at the price of a smaller n.
    """
    per_method = [
        {sid for sid, o in per_system.items() if isinstance(o, Solved)}
        for per_system in outcomes.values()
    ]
    return frozenset.intersection(*map(frozenset, per_method)) if per_method else frozenset()
