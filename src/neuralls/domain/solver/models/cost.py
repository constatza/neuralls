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


@dataclass(frozen=True, slots=True)
class OfflineStages:
    """The one-off costs paid once per method artifact, before any workload runs.

    ``generation`` is building the dataset the artifact was fit on and ``training`` is fitting
    the network or basis. A stage the method does not have is `NotApplicable` (a classical
    method has neither; a method fit inline during setup has no separate training, because
    that fit is already part of its measured setup), so a method's stages are always both
    stated and a missing number is `Unavailable`, never an absent field.
    """

    generation: StageCost
    training: StageCost

    def total(self) -> StageCost:
        """Both stages as one cost: seconds add, peak memory is the largest, any unknown stage wins.

        The result is `Unavailable` when either stage is, naming each such stage, because a sum
        with an unknown term has no value. Otherwise it is `Known` over the stages that apply
        (`MEASURED` only if every one of them was timed live) or `NotApplicable` when neither
        applies. Peak memory takes the maximum because the stages never shared a process.
        """
        named = (("generation", self.generation), ("training", self.training))
        reasons: list[str] = []
        recorded: list[Known] = []
        for name, stage in named:
            match stage:
                case Unavailable(reason=reason):
                    reasons.append(f"{name}: {reason}")
                case Known() as known:
                    recorded.append(known)
                case NotApplicable():
                    pass
        if reasons:
            return Unavailable("; ".join(reasons))
        if not recorded:
            return NotApplicable()
        memories = [
            known.peak_memory_bytes for known in recorded if known.peak_memory_bytes is not None
        ]
        live = all(known.provenance is CostProvenance.MEASURED for known in recorded)
        return Known(
            sum(known.wall_time_seconds for known in recorded),
            max(memories) if memories else None,
            CostProvenance.MEASURED if live else CostProvenance.HISTORICAL,
        )


NO_OFFLINE_COST = OfflineStages(generation=NotApplicable(), training=NotApplicable())
"""A method with nothing to generate or fit: its total cost is its online cost."""
