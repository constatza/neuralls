"""The closed outcome of one planned system, derived once from already-stored columns.

``solve_attempts`` stores two status columns per solve, ``solve_status`` and ``residual_status``.
Together they say what became of the solve; no iteration count is consulted, because zero
iterations is a legitimate count for a solved zero right-hand side and for an exhausted
``max_iterations = 0``.

| ``solve_status`` | ``residual_status`` | Outcome          | Why                                              |
|------------------|---------------------|------------------|--------------------------------------------------|
| ``converged``    | any                 | ``CONVERGED``    | the solver reached its tolerance; a zero RHS has |
|                  |                     |                  | no relative residual, so it converges unmeasured |
| ``failed``       | ``measured``        | ``UNCONVERGED``  | the solver returned an iterate that missed the   |
|                  |                     |                  | tolerance, so the true residual was recomputed   |
| ``failed``       | ``unmeasured``      | ``SOLVER_ERROR`` | the solver raised or returned a non-finite       |
|                  |                     |                  | iterate, so there was nothing to measure (a zero |
|                  |                     |                  | RHS converges and cannot fail)                   |

``SETUP_FAILED`` and ``NOT_ATTEMPTED`` have no solve row. The join of the planned systems with
the setup events decides them: a planned system with no solve row is ``SETUP_FAILED`` when its
matrix's setup failed and ``NOT_ATTEMPTED`` otherwise.
``validate_decoded_solve_frame`` rejects stored rows that contradict the table above.

The observation view (``observations``) joins the three stored frames of one method into one row
per planned system. It is computed in memory from the stored parquets and never persisted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import assert_never

import polars as pl

from neuralls.domain.solver.reporting.event_frames import validate_setup_solve_frame_links
from neuralls.domain.solver.true_residual import stopping_threshold_rel
from neuralls.shared.types import PreconditionerName


class Outcome(StrEnum):
    """What became of one planned system.

    ``UNCONVERGED``: the solver returned an iterate that missed the tolerance, either because
    the iteration cap was reached or because it flagged a breakdown yet still returned a finite
    iterate (breakdown is not stored). The iteration count is where it stopped; when the cap was
    hit it is a lower bound on the iterations needed.
    """

    CONVERGED = "converged"
    UNCONVERGED = "unconverged"
    SOLVER_ERROR = "solver_error"
    SETUP_FAILED = "setup_failed"
    NOT_ATTEMPTED = "not_attempted"


class SolveStatus(StrEnum):
    """Stored values of the ``solve_status`` column."""

    CONVERGED = "converged"
    FAILED = "failed"


class SetupStatus(StrEnum):
    """Stored values of the ``setup_status`` column."""

    COMPLETED = "completed"
    FAILED = "failed"


class ResidualStatus(StrEnum):
    """Stored values of the ``residual_status`` column."""

    MEASURED = "measured"
    UNMEASURED = "unmeasured"


def solve_outcome(solve_status: str, residual_status: str) -> Outcome:
    """Derive the outcome of one stored solve row.

    Raises:
        ValueError: A status is not one the frame schema stores.
    """
    match SolveStatus(solve_status), ResidualStatus(residual_status):
        case SolveStatus.CONVERGED, ResidualStatus.MEASURED | ResidualStatus.UNMEASURED:
            return Outcome.CONVERGED
        case SolveStatus.FAILED, ResidualStatus.MEASURED:
            return Outcome.UNCONVERGED
        case SolveStatus.FAILED, ResidualStatus.UNMEASURED:
            return Outcome.SOLVER_ERROR
        case unreachable:
            assert_never(unreachable)


_STATUS_KEY_SEPARATOR = "/"
_OUTCOME_BY_STATUS_KEY: dict[str, str] = {
    f"{solve}{_STATUS_KEY_SEPARATOR}{residual}": solve_outcome(solve, residual).value
    for solve, residual in product(SolveStatus, ResidualStatus)
}


def solve_outcome_expr() -> pl.Expr:
    """Expression giving each ``solve_attempts`` row's outcome, equal to ``solve_outcome``.

    The lookup table is built from ``solve_outcome``, so the two cannot disagree. A status pair
    the schema does not store raises when the expression is evaluated.
    """
    status_key = pl.concat_str(
        pl.col("solve_status"), pl.col("residual_status"), separator=_STATUS_KEY_SEPARATOR
    )
    return status_key.replace_strict(_OUTCOME_BY_STATUS_KEY, return_dtype=pl.String)


OBSERVATION_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "dataset": pl.String,
    "recipe": pl.String,
    "matrix_id": pl.Int64,
    "system_id": pl.String,
    "rhs_ordinal": pl.Int64,
    "method": pl.String,
    "outcome": pl.String,
    "iterations": pl.Int64,
    "solve_s": pl.Float64,
    "setup_s": pl.Float64,
    "planned_rhs": pl.Int64,
    "online_s": pl.Float64,
    "standalone_s": pl.Float64,
    "time_per_iteration_s": pl.Float64,
    "true_residual_rel": pl.Float64,
    "stopping_threshold_rel": pl.Float64,
    "timing_spread": pl.Float64,
}
"""One row per planned system of a method; ``outcome`` holds ``Outcome`` values."""

_OBSERVATION_SORT_KEYS = ("method", "dataset", "matrix_id", "rhs_ordinal", "recipe", "system_id")
_MATRIX_KEYS = ["dataset", "matrix_id"]
_SYSTEM_IDENTITY = ["dataset", "recipe", "matrix_id", "rhs_ordinal"]
_OUTCOMES_WITH_ITERATIONS = [Outcome.CONVERGED.value, Outcome.UNCONVERGED.value]


@dataclass(frozen=True, slots=True, eq=False)
class MethodFrames:
    """The three stored frames of one method, as its parquet artifacts hold them."""

    planned_systems: pl.DataFrame
    setup_events: pl.DataFrame
    solve_attempts: pl.DataFrame


def observations_of_methods(per_method: Mapping[PreconditionerName, MethodFrames]) -> pl.DataFrame:
    """The observation view of several methods, sorted by method, dataset, matrix and RHS."""
    views = [
        observations(method, frames.planned_systems, frames.setup_events, frames.solve_attempts)
        for method, frames in per_method.items()
    ]
    if not views:
        return pl.DataFrame(schema=OBSERVATION_SCHEMA)
    return pl.concat(views).sort(_OBSERVATION_SORT_KEYS)


def observations(
    method: PreconditionerName,
    planned_systems: pl.DataFrame,
    setup_events: pl.DataFrame,
    solve_attempts: pl.DataFrame,
) -> pl.DataFrame:
    """Join one method's stored frames into one row per planned system.

    ``planned_rhs`` counts the planned systems of a system's ``(dataset, matrix_id)`` over every
    recipe in ``planned_systems``, which is the count the amortized online time divides the
    matrix setup by.

    Raises:
        ValueError: A frame belongs to another method, a solve row is not a planned system, or
            a solve row contradicts its planned system or its setup event.
    """
    _validate_frames(method, planned_systems, setup_events, solve_attempts)
    planned = planned_systems.select("system_id", *_SYSTEM_IDENTITY, "rtol", "atol").with_columns(
        planned_rhs=pl.len().over(_MATRIX_KEYS).cast(pl.Int64)
    )
    solves = solve_attempts.select(
        "system_id",
        "solve_s",
        "iterations",
        "true_residual_rel",
        "rhs_norm",
        "samples_s",
        solve_outcome=solve_outcome_expr(),
    )
    setups = setup_events.select(*_MATRIX_KEYS, "setup_s", "setup_status")
    joined = (
        planned.join(solves, on="system_id", how="left", validate="1:1")
        .join(setups, on=_MATRIX_KEYS, how="left", validate="m:1")
        .with_columns(
            outcome=pl.coalesce(
                pl.col("solve_outcome"),
                pl.when(pl.col("setup_status") == SetupStatus.FAILED.value)
                .then(pl.lit(Outcome.SETUP_FAILED.value))
                .otherwise(pl.lit(Outcome.NOT_ATTEMPTED.value)),
            )
        )
    )
    has_iterations = pl.col("outcome").is_in(_OUTCOMES_WITH_ITERATIONS)
    spread = _timing_spread_expr()
    return (
        joined.with_columns(
            method=pl.lit(str(method), dtype=pl.String),
            iterations=pl.when(has_iterations).then(pl.col("iterations")),
        )
        .with_columns(
            online_s=pl.col("solve_s") + pl.col("setup_s") / pl.col("planned_rhs"),
            standalone_s=pl.col("solve_s") + pl.col("setup_s"),
            time_per_iteration_s=pl.when(pl.col("iterations") >= 1).then(
                pl.col("solve_s") / pl.col("iterations")
            ),
            stopping_threshold_rel=_stopping_thresholds(joined),
            timing_spread=spread,
        )
        .select(list(OBSERVATION_SCHEMA))
        .cast(pl.Schema(OBSERVATION_SCHEMA))
        .sort(_OBSERVATION_SORT_KEYS)
    )


def _timing_spread_expr() -> pl.Expr:
    """``(max - min) / median`` of the timing repeats; undefined for a zero median."""
    samples = pl.col("samples_s").list
    median = samples.median()
    return pl.when(median > 0).then((samples.max() - samples.min()) / median)


def _stopping_thresholds(joined: pl.DataFrame) -> pl.Series:
    """Relative stopping level of each system; null without a nonzero measured RHS norm."""
    thresholds = [
        stopping_threshold_rel(rtol, atol, rhs_norm) if rhs_norm else None
        for rtol, atol, rhs_norm in zip(
            joined["rtol"], joined["atol"], joined["rhs_norm"], strict=True
        )
    ]
    return pl.Series("stopping_threshold_rel", thresholds, dtype=pl.Float64)


def _validate_frames(
    method: PreconditionerName,
    planned_systems: pl.DataFrame,
    setup_events: pl.DataFrame,
    solve_attempts: pl.DataFrame,
) -> None:
    """Reject frames the join would silently mis-attribute."""
    for name, frame in (("setup events", setup_events), ("solve attempts", solve_attempts)):
        if (frame["method"] != str(method)).any():
            raise ValueError(f"{name} contain rows of a method other than {method!r}")
    if planned_systems["system_id"].is_duplicated().any():
        raise ValueError("duplicate planned system ID")
    if setup_events.select(_MATRIX_KEYS).is_duplicated().any():
        raise ValueError("duplicate setup event for one dataset and matrix")
    if solve_attempts["system_id"].is_duplicated().any():
        raise ValueError("duplicate solve attempt for one system")
    validate_setup_solve_frame_links(setup_events, solve_attempts)
    solved = solve_attempts.select("system_id", *_SYSTEM_IDENTITY).join(
        planned_systems.select("system_id", *_SYSTEM_IDENTITY),
        on="system_id",
        how="left",
        suffix="_planned",
    )
    if solved["dataset_planned"].is_null().any():
        raise ValueError("solve attempt is not a planned system")
    for column in _SYSTEM_IDENTITY:
        if (solved[column] != solved[f"{column}_planned"]).any():
            raise ValueError(f"solve attempt {column} does not match its planned system")
