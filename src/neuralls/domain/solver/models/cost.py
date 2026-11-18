"""One pipeline stage's wall-clock cost and where the number came from."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

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


class FitFolds(StrEnum):
    """The folds of a dataset's pinned split that the job fitting an artifact read.

    A POD basis, fit inside the comparison, reads only the train fold. A neural
    training or search job also reads the validation fold, for early stopping and checkpoint
    selection, so those rows served the artifact too.
    """

    TRAIN = "train"
    TRAIN_AND_VALIDATION = "train_and_validation"


@dataclass(frozen=True, slots=True)
class TrainShare:
    """The fraction of a dataset's rows that a fit read, in (0, 1], and the folds they are.

    A fit that read no row has no share: nothing was fit, so there is nothing to charge.
    """

    value: float
    folds: FitFolds

    def __post_init__(self) -> None:
        """Reject a share outside (0, 1], NaN included."""
        if not 0.0 < self.value <= 1.0:
            raise ValueError(f"a train share lies in (0, 1], got {self.value}")

    @classmethod
    def of_rows(cls, read_rows: int, total_rows: int, folds: FitFolds) -> TrainShare:
        """The share ``read_rows / total_rows`` of ``folds``; raises when no row was read."""
        if not 0 < read_rows <= total_rows:
            raise ValueError(f"{read_rows} read rows of {total_rows} have no train share")
        return cls(read_rows / total_rows, folds)


@dataclass(frozen=True, slots=True)
class TrainShareOfGeneration:
    """The share of a dataset's recorded generation charged to an artifact fit on it.

    Generation produced every row of the dataset, but a fit reads only the folds of its pinned
    split that ``share`` names, so the artifact is charged ``dataset`` scaled by ``share``; the
    held-out test rows serve evaluation. Both numbers are kept, so the whole generation and the
    share charged stay stated.
    """

    dataset: Known
    share: TrainShare

    def charged(self) -> Known:
        """The seconds charged to the artifact; peak memory is the whole generation's."""
        return Known(
            self.dataset.wall_time_seconds * self.share.value,
            self.dataset.peak_memory_bytes,
            self.dataset.provenance,
        )


type GenerationCost = NotApplicable | Unavailable | TrainShareOfGeneration
"""The generation stage: every fit reads only some folds of its dataset's pinned split, so a
recorded generation is always charged as a `TrainShareOfGeneration`."""


def recorded_seconds(cost: StageCost) -> float | None:
    """Wall time of a recorded cost; None for a stage that is not applicable or unavailable."""
    match cost:
        case Known(wall_time_seconds=seconds):
            return seconds
        case NotApplicable() | Unavailable():
            return None


@dataclass(frozen=True, slots=True)
class NoFit:
    """The method has no fitted artifact: neither a dataset to generate nor a fit to run."""


@dataclass(frozen=True, slots=True)
class InlineFit:
    """The method is fit from its dataset inside the comparison, once, before its timed setups.

    The fit is measured as it runs, so its fit stage is `Known` with `MEASURED` provenance.
    """


@dataclass(frozen=True, slots=True)
class FitJob:
    """The method's artifact was fit ahead of time by a job whose MLflow run is the fit stage.

    Its fit stage is charged back from that run (`HISTORICAL`), or `Unavailable` when the run
    recorded no duration.
    """


type FitOrigin = NoFit | InlineFit | FitJob


def _origin_of_known_fit(provenance: CostProvenance) -> InlineFit | FitJob:
    match provenance:
        case CostProvenance.MEASURED:
            return InlineFit()
        case CostProvenance.HISTORICAL:
            return FitJob()
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class OfflineStages:
    """The one-off costs paid once per method artifact, before any workload runs.

    ``generation`` is building the dataset the artifact was fit on (only the share of it in the
    folds the fit read) and ``fit`` is fitting the network or basis: the measured fit a POD
    method runs once inside the comparison, before its timed setups, or the training job that
    produced a checkpoint. Together they are the ``offline`` cost of the vocabulary in
    ``domain/solver/README.md``. A stage the method does not have is `NotApplicable` (a
    classical method has neither), so a method's stages are always both stated and a missing
    number is `Unavailable`, never an absent field.
    """

    generation: GenerationCost
    fit: StageCost

    def charged_generation(self) -> StageCost:
        """The generation cost charged to the artifact, as a plain stage cost."""
        match self.generation:
            case TrainShareOfGeneration() as share:
                return share.charged()
            case NotApplicable() | Unavailable() as stage:
                return stage
            case unreachable:
                assert_never(unreachable)

    def fit_origin(self) -> FitOrigin:
        """Where the method's fit happened, read from the fit stage's provenance.

        A fit measured live ran inside the comparison; one charged back from history, or one
        whose job recorded no duration, was a fit job. A method with no fit stage has nothing
        to fit.
        """
        match self.fit:
            case Known(provenance=provenance):
                return _origin_of_known_fit(provenance)
            case Unavailable():
                return FitJob()
            case NotApplicable():
                return NoFit()
            case unreachable:
                assert_never(unreachable)

    def combined(self) -> StageCost:
        """Both stages as one cost: seconds add, peak memory is the largest, any unknown stage wins.

        The result is `Unavailable` when either stage is, naming each such stage, because a sum
        with an unknown term has no value. Otherwise it is `Known` over the stages that apply
        (`MEASURED` only if every one of them was timed live) or `NotApplicable` when neither
        applies. Peak memory takes the maximum because the stages never shared a process.
        """
        named = (("generation", self.charged_generation()), ("fit", self.fit))
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


NO_OFFLINE_COST = OfflineStages(generation=NotApplicable(), fit=NotApplicable())
"""A method with nothing to generate or fit: its full cost is its online cost."""
