"""Total cost of a method: its offline stages plus its online workload.

Offline cost is paid once per method artifact and online cost once per workload, so a total is
only meaningful for one stated workload. A method whose offline cost is not recorded has no
total: reporting its online cost as the total would read the unknown stage as free.
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


type TotalCost = Complete | OfflineUnavailable


def total_cost_seconds(offline: OfflineStages, workload_online_s: float) -> TotalCost:
    """The total cost of a method over a workload that took ``workload_online_s`` online.

    ``workload_online_s`` is the workload's setup plus solve seconds. An offline cost that
    does not apply adds nothing; one that applies but is unavailable gives `OfflineUnavailable`.
    """
    match offline.total():
        case NotApplicable():
            return Complete(workload_online_s)
        case Known(wall_time_seconds=offline_s):
            return Complete(offline_s + workload_online_s)
        case Unavailable():
            return OfflineUnavailable(workload_online_s)
