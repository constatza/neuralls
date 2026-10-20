"""Profiles of how many systems each preconditioner solves, built from `SystemOutcome`s.

Both profiles are cumulative distributions over systems in which a failed run
stays an unsolved system. Nothing is averaged and no penalty is invented:

- Performance profile (Dolan & Moré 2002, "Benchmarking optimization software
  with performance profiles", Math. Program. 91): the fraction of systems on
  which a method's cost is within a factor tau of the best method's cost on
  that system. A failure is ratio infinity.
- ECDF profile (Hansen et al. 2016, "COCO: Performance Assessment",
  arXiv:1605.03560): the fraction of systems solved within an absolute budget
  of iterations or seconds. It does not depend on which other methods are
  compared. (Moré & Wild's 2009 data profile additionally divides the budget by
  n_p + 1; this one does not.)

Gould & Scott (2016, ACM TOMS 43) caution that a profile is not an estimate of
expected performance, so these are shown beside explicit success counts.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.domain.benchmarking.costs import Cost
from neuralls.domain.solver.models.outcomes import Failed, Outcomes, Solved, SystemOutcome


@dataclass(frozen=True, slots=True)
class Profile:
    """One method's cumulative distribution over the systems it was run on.

    Attributes:
        n_systems: Systems counted, failures included.
        solved_at: Ascending values at which a system was solved (a cost
            ratio for a performance profile, a budget for an ECDF profile).
            Failures contribute no value, so the curve plateaus below 1.
    """

    n_systems: int
    solved_at: tuple[float, ...]

    def fraction_solved(self, x: float) -> float:
        """Fraction of systems solved at a value of at most ``x``."""
        return bisect.bisect_right(self.solved_at, x) / self.n_systems


def _cost_of(outcome: SystemOutcome, cost: Cost) -> float | Failed:
    """A run's numeric cost, or its censored failure state.

    Zero is a valid cost (a run converged at the initial guess takes 0
    iterations); `performance_profiles` skips systems whose best cost is zero
    because a ratio to it is undefined.
    """
    match outcome:
        case Failed():
            return outcome
        case Solved():
            return cost(outcome)


def _costs(outcomes: Outcomes, cost: Cost) -> dict[str, dict[str, float | Failed]]:
    """Each solved cost or censored failure, preserving every observation."""
    return {
        name: {sid: _cost_of(outcome, cost) for sid, outcome in per_system.items()}
        for name, per_system in outcomes.items()
    }


def ecdf_profiles(outcomes: Outcomes, cost: Cost) -> dict[str, Profile]:
    """Per method, the fraction of its systems solved within each absolute budget.

    The unnormalised empirical CDF of runtimes (Hansen et al. 2016, "COCO:
    Performance Assessment", arXiv:1605.03560). It is not the Moré & Wild
    (2009) data profile, which divides the budget by n_p + 1 simplex
    gradients. It does not depend on which other methods are compared.
    """
    return {
        name: Profile(
            n_systems=len(per_system),
            solved_at=tuple(sorted(v for v in per_system.values() if not isinstance(v, Failed))),
        )
        for name, per_system in _costs(outcomes, cost).items()
        if per_system
    }


def performance_profiles(outcomes: Outcomes, cost: Cost) -> dict[str, Profile]:
    """Per method, the fraction of systems within each factor of the best method's cost.

    Only systems that at least one method solved have a best cost; the others
    are excluded for every method, as Dolan & Moré do.
    """
    costs = _costs(outcomes, cost)
    best = _best_cost_by_system(costs)
    profiles: dict[str, Profile] = {}
    for name, per_system in costs.items():
        scored = {sid: v for sid, v in per_system.items() if best.get(sid, 0.0) > 0.0}
        if not scored:
            continue
        ratios = sorted(
            value / best[sid] for sid, value in scored.items() if not isinstance(value, Failed)
        )
        profiles[name] = Profile(n_systems=len(scored), solved_at=tuple(ratios))
    return profiles


def _best_cost_by_system(costs: Mapping[str, Mapping[str, float | Failed]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for per_system in costs.values():
        for system_id, value in per_system.items():
            if not isinstance(value, Failed):
                best[system_id] = min(value, best.get(system_id, value))
    return best
