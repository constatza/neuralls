"""What a group figure's card records about the evidence its numbers rest on.

Each record is a closed variant so a card never carries a field whose meaning depends on another:
an interval is either stated in full or absent, the RHS counts that share a matrix's setup exist
only for a metric that divides setup over them, and timing evidence exists only for a metric
measured in seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.breakeven_full import full_comparisons
from neuralls.domain.benchmarking.paired import compared_methods, paired_systems, speed_ratios
from neuralls.domain.benchmarking.uncertainty import (
    BootstrapKind,
    ClusterBootstrap,
    TooFewMatrices,
    cluster_bootstrap,
)
from neuralls.domain.solver.models.cost import OfflineStages
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId, Unit
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.summary import (
    Population,
    outcome_counts,
    select,
    timing_noise,
)
from neuralls.domain.solver.reporting.timed_values import bootstrap_kind_of
from neuralls.shared.types import PreconditionerName

SINGLE_REPEAT_WARNING = (
    "each solve was timed once, so its time carries unmeasured noise that this figure cannot show"
)
"""The card's wording when a timing figure rests on one timing repeat per solve."""

_SINGLE_REPEAT = 1


@dataclass(frozen=True, slots=True)
class NotDrawnByKind:
    """The figure's statistic never draws an interval."""


@dataclass(frozen=True, slots=True)
class MethodInterval:
    """The interval of one method's mark over that method's own matrices, or why it has none."""

    method: PreconditionerName
    interval: TooFewMatrices | ClusterBootstrap


@dataclass(frozen=True, slots=True)
class PerMethodIntervals:
    """The figure draws an interval per method, each resampling only the matrices of its method.

    ``methods`` is ordered by method name and names each method once.
    """

    methods: tuple[MethodInterval, ...]

    def __post_init__(self) -> None:
        """Require each method once, in name order; a cohort with no systems has none."""
        names = [entry.method for entry in self.methods]
        if names != sorted(set(names)):
            raise ValueError("per-method intervals must name each method once, in name order")


@dataclass(frozen=True, slots=True)
class PairInterval:
    """The interval of one method against one compared method, over the matrices of their pairs."""

    method: PreconditionerName
    reference_method: PreconditionerName
    interval: TooFewMatrices | ClusterBootstrap


@dataclass(frozen=True, slots=True)
class PerPairIntervals:
    """The figure draws an interval per ordered method pair, each over that pair's own matrices.

    ``pairs`` is ordered by method, then compared method, and names each ordered pair once.
    """

    pairs: tuple[PairInterval, ...]

    def __post_init__(self) -> None:
        """Require each ordered pair once, in name order."""
        keys = [(entry.method, entry.reference_method) for entry in self.pairs]
        if keys != sorted(set(keys)):
            raise ValueError("per-pair intervals must name each ordered pair once, in name order")


type FigureInterval = NotDrawnByKind | PerMethodIntervals | PerPairIntervals
"""The intervals a figure draws, or that its statistic never draws one."""


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


def intervals_of_speed_ratios(
    observations: pl.DataFrame, reference: PreconditionerName, metric: MetricId
) -> PerMethodIntervals:
    """The bootstrap behind each method's speed-ratio interval against ``reference``.

    A method's count is the distinct matrices that give it a ratio, the clusters
    ``speed_ratio_summary`` resamples for that method.
    """
    ratios = speed_ratios(paired_systems(observations, reference, metric))
    return _per_method(ratios, compared_methods(observations, reference), bootstrap_kind_of(metric))


def intervals_of_means(
    observations: pl.DataFrame, metric: MetricId, population: Population
) -> PerMethodIntervals:
    """The bootstrap behind each method's whisker of its mean of ``metric`` over ``population``.

    A method's count is the distinct matrices that give it a defined value, the clusters
    ``summarize`` resamples for that method.
    """
    methods = sorted(observations["method"].unique().to_list())
    return _per_method(select(observations, metric, population), methods, bootstrap_kind_of(metric))


def _per_method(
    rows: pl.DataFrame, methods: Sequence[str], kind: BootstrapKind
) -> PerMethodIntervals:
    """The ``kind`` interval of each of ``methods`` over the distinct matrices its ``rows`` cover."""
    counts = dict(
        rows.group_by("method")
        .agg(pl.struct("dataset", "matrix_id").n_unique().alias("matrices"))
        .iter_rows()
    )
    return PerMethodIntervals(
        tuple(
            MethodInterval(
                PreconditionerName(method), cluster_bootstrap(counts.get(method, 0), kind)
            )
            for method in methods
        )
    )


def intervals_of_breakeven_full(
    observations: pl.DataFrame, offline: Mapping[PreconditionerName, OfflineStages]
) -> PerPairIntervals:
    """The two-level bootstrap behind each method pair's offline break-even interval.

    A pair's count is the distinct matrices of the pairs both methods converged, the clusters
    ``breakeven_full`` resamples for it.
    """
    return PerPairIntervals(
        tuple(
            PairInterval(
                comparison.method,
                comparison.reference_method,
                cluster_bootstrap(comparison.n_matrices, bootstrap_kind_of(MetricId.ONLINE_S)),
            )
            for comparison in full_comparisons(observations, offline)
        )
    )


def setup_sharing_of(metric: MetricId, observations: pl.DataFrame) -> SetupSharing:
    """The right-hand-side counts that divide a matrix's setup in ``metric``, if it divides one."""
    match metric:
        case MetricId.ONLINE_AMORTIZED_S:
            per_matrix = observations.unique(["dataset", "matrix_id"])["planned_rhs"]
            return SetupSharedOverRhs(tuple(sorted(per_matrix.unique().to_list())))
        case (
            MetricId.ITERATIONS
            | MetricId.SOLVE_S
            | MetricId.ONLINE_S
            | MetricId.SETUP_S
            | MetricId.TIME_PER_ITERATION_S
            | MetricId.CONVERGENCE_FACTOR
        ):
            return NoSharedSetup()
        case unreachable:
            assert_never(unreachable)


def _is_time(metric: MetricId) -> bool:
    match METRICS[metric].unit:
        case Unit.ITERATIONS | Unit.REDUCTION_PER_ITERATION:
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
