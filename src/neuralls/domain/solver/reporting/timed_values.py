"""How each metric's value is composed from the timed events behind it, for its interval.

A metric measured in seconds is, on every observation row, ``a * t_setup + b * t_solve``: the
representative times of the row's matrix setup and of its solve, weighted by the row (the
amortized online time divides the setup by the matrix's planned RHS, the time per iteration
divides the solve by the iterations). Its interval is the two-level bootstrap, which redraws
both events' stored timing repeats. Every other metric is deterministic per pair and keeps the
matrix-cluster bootstrap.

The observation view carries each row's repeats as ``setup_samples_s`` and ``solve_samples_s``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import assert_never

import numpy as np
import polars as pl

from neuralls.domain.benchmarking.uncertainty import BootstrapKind, MatrixCluster, TimedCluster
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import MatrixId

SETUP_SAMPLES = "setup_samples_s"
SOLVE_SAMPLES = "solve_samples_s"
_SETUP_WEIGHT = "setup_weight"
_SOLVE_WEIGHT = "solve_weight"
_MATRIX_KEYS = ["dataset", "matrix_id"]
REFERENCE_SUFFIX = "_reference"


@dataclass(frozen=True, slots=True)
class TimedMetric:
    """A metric in seconds: each row is ``setup_weight * t_setup + solve_weight * t_solve``."""

    setup_weight: pl.Expr
    solve_weight: pl.Expr


ONLINE_TIMING = TimedMetric(pl.lit(1.0), pl.lit(1.0))
"""The pair online time: its matrix's whole setup plus its solve."""


@dataclass(frozen=True, slots=True)
class UntimedMetric:
    """A metric that is deterministic per pair, so it carries no timing noise."""


def timing_of(metric: MetricId) -> TimedMetric | UntimedMetric:
    """How ``metric``'s value on a row is composed from the row's setup and solve times."""
    match metric:
        case MetricId.SOLVE_S:
            return TimedMetric(pl.lit(0.0), pl.lit(1.0))
        case MetricId.ONLINE_S:
            return ONLINE_TIMING
        case MetricId.ONLINE_AMORTIZED_S:
            return TimedMetric(1.0 / pl.col("planned_rhs"), pl.lit(1.0))
        case MetricId.SETUP_S:
            return TimedMetric(pl.lit(1.0), pl.lit(0.0))
        case MetricId.TIME_PER_ITERATION_S:
            return TimedMetric(pl.lit(0.0), 1.0 / pl.col("iterations"))
        case MetricId.ITERATIONS | MetricId.CONVERGENCE_FACTOR:
            return UntimedMetric()
        case unreachable:
            assert_never(unreachable)


def bootstrap_kind_of(metric: MetricId) -> BootstrapKind:
    """The bootstrap behind ``metric``'s intervals: two-level for a time, else matrix-cluster."""
    match timing_of(metric):
        case TimedMetric():
            return BootstrapKind.TWO_LEVEL
        case UntimedMetric():
            return BootstrapKind.MATRIX_CLUSTER
        case unreachable:
            assert_never(unreachable)


TIMING_COLUMNS = (SETUP_SAMPLES, SOLVE_SAMPLES, _SETUP_WEIGHT, _SOLVE_WEIGHT)
"""The columns ``with_timing_weights`` leaves on a row for its events to be redrawn."""


def with_timing_weights(rows: pl.DataFrame, timing: TimedMetric) -> pl.DataFrame:
    """``rows`` with the weight of each row's setup and solve time in its value."""
    return rows.with_columns(
        timing.setup_weight.cast(pl.Float64).alias(_SETUP_WEIGHT),
        timing.solve_weight.cast(pl.Float64).alias(_SOLVE_WEIGHT),
    )


def timed_clusters(rows: pl.DataFrame, timing: TimedMetric) -> dict[MatrixCluster, TimedCluster]:
    """Each matrix of ``rows`` (one method's selected rows), its values linear in its events.

    Raises:
        ValueError: A row that reads an event has no stored repeats for it.
    """
    clusters: dict[MatrixCluster, TimedCluster] = {}
    weighted = with_timing_weights(rows, timing)
    for (dataset, matrix_id), group in weighted.group_by(_MATRIX_KEYS, maintain_order=True):
        events: list[tuple[float, ...]] = []
        dense = _dense(_side_weights(group.to_dicts(), events), len(events))
        clusters[(_dataset(dataset), MatrixId(_matrix(matrix_id)))] = TimedCluster(
            tuple(events), lambda times, dense=dense: times @ dense.T
        )
    return clusters


def timed_ratio_clusters(pairs: pl.DataFrame) -> dict[MatrixCluster, TimedCluster]:
    """Each matrix of ``pairs``, its reference-over-method ratios as functions of their events.

    ``pairs`` holds one row per system with the ``TIMING_COLUMNS`` of the method and, suffixed
    ``_reference``, of the reference; a ratio redraws both methods' events of its matrix.

    Raises:
        ValueError: A row that reads an event has no stored repeats for it.
    """
    return _timed_pair_clusters(pairs, np.divide)


def timed_difference_clusters(pairs: pl.DataFrame) -> dict[MatrixCluster, TimedCluster]:
    """Each matrix of ``pairs``, its reference-minus-method differences as functions of events.

    ``pairs`` is shaped as for ``timed_ratio_clusters``.

    Raises:
        ValueError: A row that reads an event has no stored repeats for it.
    """
    return _timed_pair_clusters(pairs, np.subtract)


def _timed_pair_clusters(
    pairs: pl.DataFrame, combine: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> dict[MatrixCluster, TimedCluster]:
    """Each matrix of ``pairs``, ``combine(reference values, method values)`` of its events."""
    clusters: dict[MatrixCluster, TimedCluster] = {}
    for (dataset, matrix_id), group in pairs.group_by(_MATRIX_KEYS, maintain_order=True):
        rows = group.to_dicts()
        events: list[tuple[float, ...]] = []
        method = _side_weights(rows, events)
        reference = _side_weights(rows, events, suffix=REFERENCE_SUFFIX)
        left, right = _dense(reference, len(events)), _dense(method, len(events))
        clusters[(_dataset(dataset), MatrixId(_matrix(matrix_id)))] = TimedCluster(
            tuple(events),
            lambda times, left=left, right=right: combine(times @ left.T, times @ right.T),
        )
    return clusters


def _side_weights(
    rows: Sequence[Mapping[str, object]], events: list[tuple[float, ...]], *, suffix: str = ""
) -> list[dict[int, float]]:
    """Each row's weight per event of one side; the side's events are appended to ``events``.

    The rows of one side share their matrix's setup, so it is one event; every solve is its own.
    """
    setup_index: int | None = None
    weights: list[dict[int, float]] = []
    for row in rows:
        row_weights: dict[int, float] = {}
        setup_weight = _weight(row[f"{_SETUP_WEIGHT}{suffix}"])
        solve_weight = _weight(row[f"{_SOLVE_WEIGHT}{suffix}"])
        if setup_weight != 0.0:
            if setup_index is None:
                setup_index = len(events)
                events.append(_samples(row[f"{SETUP_SAMPLES}{suffix}"], "setup"))
            row_weights[setup_index] = setup_weight
        if solve_weight != 0.0:
            row_weights[len(events)] = solve_weight
            events.append(_samples(row[f"{SOLVE_SAMPLES}{suffix}"], "solve"))
        weights.append(row_weights)
    return weights


def _dense(weights: Sequence[Mapping[int, float]], n_events: int) -> np.ndarray:
    dense = np.zeros((len(weights), n_events))
    for row, row_weights in enumerate(weights):
        for event, weight in row_weights.items():
            dense[row, event] = weight
    return dense


def _weight(value: object) -> float:
    if not isinstance(value, float):
        raise TypeError("a timed observation needs a numeric weight for each event")
    return value


def _samples(value: object, event: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or not value:
        raise ValueError(f"a timed observation has no stored {event} repeats")
    return tuple(float(sample) for sample in value)


def _dataset(value: object) -> str | None:
    return None if value is None else str(value)


def _matrix(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("a matrix cluster needs an integer matrix ID")
    return value
