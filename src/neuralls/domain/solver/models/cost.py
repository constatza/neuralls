"""One pipeline stage's wall-clock cost and where the number came from."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.types import CostProvenance


@dataclass(frozen=True, slots=True)
class NotApplicable:
    """The stage does not exist for this method (no training data, no fit step).

    Reads as zero wherever costs are totalled; distinct from `Unavailable`, which is a
    stage that exists but whose cost was never recorded.
    """


@dataclass(frozen=True, slots=True)
class Known:
    """A recorded stage cost, tagged with where the number came from.

    See shared.types.CostProvenance for what MEASURED/HISTORICAL mean.
    """

    wall_time_seconds: float
    peak_memory_bytes: int | None
    provenance: CostProvenance


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A stage that applies to the method but whose cost was not recorded.

    Carries no number, so it can never be read as a zero cost.
    """

    reason: str


type StageCost = NotApplicable | Known | Unavailable


def recorded_seconds(cost: StageCost) -> float | None:
    """Wall time of a recorded cost; None for a stage that is not applicable or unavailable."""
    match cost:
        case Known(wall_time_seconds=seconds):
            return seconds
        case NotApplicable() | Unavailable():
            return None
