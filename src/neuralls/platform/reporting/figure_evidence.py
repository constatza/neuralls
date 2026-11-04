"""What a group figure's card records about the evidence its numbers rest on.

Each record is a closed variant so a card never carries a field whose meaning depends on another:
an interval is either stated in full or absent, the RHS counts that share a matrix's setup exist
only for a metric that divides setup over them, and timing evidence exists only for a metric
measured in seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.paired import paired_systems, speed_ratios
from neuralls.domain.benchmarking.uncertainty import MatrixClusterBootstrap
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId, Unit
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.summary import outcome_counts, timing_noise
from neuralls.shared.types import PreconditionerName

SINGLE_REPEAT_WARNING = (
    "each solve was timed once, so its time carries unmeasured noise that this figure cannot show"
)
"""The card's wording when a timing figure rests on one timing repeat per solve."""

_SINGLE_REPEAT = 1


@dataclass(frozen=True, slots=True)
class NoInterval:
    """The figure draws no interval."""


type FigureInterval = NoInterval | MatrixClusterBootstrap


@dataclass(frozen=True, slots=True)
class NoSharedSetup:
    """The figure's metric does not divide a matrix's setup over its right-hand sides."""


@dataclass(frozen=True, slots=True)
class SetupSharedOverRhs:
    """The figure's metric divides each matrix's setup over its planned right-hand sides.

    ``rhs_counts`` holds the distinct counts $r_m$ over the matrices of the figure, ascending.
    """

    rhs_counts: tuple[int, ...]


type SetupSharing = NoSharedSetup | SetupSharedOverRhs


@dataclass(frozen=True, slots=True)
class MethodOutcomeCounts:
    """The number of a method's planned systems that ended in each outcome."""

    method: PreconditionerName
    counts: Mapping[Outcome, int]

    def __post_init__(self) -> None:
        """Require one non-negative count for every outcome."""
        if set(self.counts) != set(Outcome):
            raise ValueError(f"outcome counts of {self.method} must give every outcome once")
        if any(count < 0 for count in self.counts.values()):
            raise ValueError(f"outcome counts of {self.method} must not be negative")


@dataclass(frozen=True, slots=True)
class MethodTimingSpread:
    """The median over a method's timed systems of ``(max - min) / median`` of their repeats."""

    method: PreconditionerName
    median_spread: float


class TimingRepeats(StrEnum):
    """Whether each timed solve was repeated, which decides if its noise can be judged."""

    REPEATED = "repeated"
    SINGLE = "single"


@dataclass(frozen=True, slots=True)
class NoTimingEvidence:
    """The figure's metric is not a time."""


@dataclass(frozen=True, slots=True)
class TimingEvidence:
    """How reliable the times behind a timing figure are.

    ``spreads`` has an entry per method with at least one timed system whose spread is defined.
    """

    spreads: tuple[MethodTimingSpread, ...]
    repeats: TimingRepeats


type FigureTimingEvidence = NoTimingEvidence | TimingEvidence


def outcome_counts_of(observations: pl.DataFrame) -> tuple[MethodOutcomeCounts, ...]:
    """The outcome counts of every method of ``observations``, ordered by method name."""
    return tuple(
        MethodOutcomeCounts(
            PreconditionerName(row["method"]),
            {outcome: int(row[outcome.value]) for outcome in Outcome},
        )
        for row in outcome_counts(observations).iter_rows(named=True)
    )


def bootstrap_of_speed_ratios(
    observations: pl.DataFrame, reference: PreconditionerName, metric: MetricId
) -> MatrixClusterBootstrap:
    """The bootstrap behind the speed-ratio intervals of every method against ``reference``.

    ``matrices`` counts the distinct matrices that give any method a ratio.
    """
    ratios = speed_ratios(paired_systems(observations, reference, metric))
    return MatrixClusterBootstrap(matrices=ratios.select("dataset", "matrix_id").unique().height)


def setup_sharing_of(metric: MetricId, observations: pl.DataFrame) -> SetupSharing:
    """The right-hand-side counts that divide a matrix's setup in ``metric``, if it divides one."""
    match metric:
        case MetricId.ONLINE_S:
            per_matrix = observations.unique(["dataset", "matrix_id"])["planned_rhs"]
            return SetupSharedOverRhs(tuple(sorted(per_matrix.unique().to_list())))
        case (
            MetricId.ITERATIONS
            | MetricId.SOLVE_S
            | MetricId.STANDALONE_S
            | MetricId.SETUP_S
            | MetricId.TIME_PER_ITERATION_S
        ):
            return NoSharedSetup()
        case unreachable:
            assert_never(unreachable)


def _is_time(metric: MetricId) -> bool:
    match METRICS[metric].unit:
        case Unit.ITERATIONS:
            return False
        case Unit.SECONDS | Unit.SECONDS_PER_ITERATION:
            return True
        case unreachable:
            assert_never(unreachable)


def timing_evidence_of(
    metric: MetricId,
    observations: pl.DataFrame,
    contexts: tuple[SolverTimingContext, ...],
) -> FigureTimingEvidence:
    """The timing evidence of a figure of ``metric``; a solve timed once anywhere is single."""
    if not _is_time(metric):
        return NoTimingEvidence()
    spreads = tuple(
        MethodTimingSpread(PreconditionerName(row["method"]), row["median_timing_spread"])
        for row in timing_noise(observations).iter_rows(named=True)
        if row["median_timing_spread"] is not None
    )
    single = any(context.timing_repeats <= _SINGLE_REPEAT for context in contexts)
    return TimingEvidence(spreads, TimingRepeats.SINGLE if single else TimingRepeats.REPEATED)
