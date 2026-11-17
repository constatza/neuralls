"""Full cost of a method: its offline stages plus its online workload over pairs.

Offline cost (``generation + fit``) is paid once per method artifact and online cost once per
workload, so a full cost is only meaningful for one stated workload. A method whose offline cost
is not recorded has no full cost: reporting its online cost as the full cost would read the
unknown stage as free. The names follow the cost vocabulary in ``domain/solver/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.solver.models.cost import (
    Known,
    NotApplicable,
    OfflineStages,
    Unavailable,
)


@dataclass(frozen=True, slots=True)
class Complete:
    """Every cost stage is known or does not apply; ``seconds`` is offline plus online."""

    seconds: float


@dataclass(frozen=True, slots=True)
class OfflineUnavailable:
    """An offline stage applies but was not recorded, so only the online seconds are known."""

    online_s: float


type FullCost = Complete | OfflineUnavailable


def full_cost_seconds(offline: OfflineStages, workload_online_s: float) -> FullCost:
    """The full cost of a method over a workload that took ``workload_online_s`` online.

    ``workload_online_s`` is the pair workload: every (matrix, RHS) pair charged its matrix's
    setup plus its solve. An offline cost that does not apply adds nothing; one that applies but
    is unavailable gives `OfflineUnavailable`.
    """
    match offline.combined():
        case NotApplicable():
            return Complete(workload_online_s)
        case Known(wall_time_seconds=offline_s):
            return Complete(offline_s + workload_online_s)
        case Unavailable():
            return OfflineUnavailable(workload_online_s)
