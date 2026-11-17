r"""Cumulative cost of solving $n$ right-hand sides of one matrix, against the reference.

The curves are the two straight lines whose crossing ``break_even`` computes, sampled at every
whole RHS count, so a break-even figure and its stored table hold the same numbers.
"""

from __future__ import annotations

import math
from enum import StrEnum

import polars as pl

from neuralls.domain.benchmarking.paired import break_even
from neuralls.shared.types import PreconditionerName

BREAK_EVEN_HORIZON_LIMIT = 1_000
"""The largest RHS count a curve is sampled at when the break-even count lies further out.

A per-RHS saving close to zero puts $n^*$ arbitrarily far away; sampling up to it would make the
table as long as $n^*$ for a line that is already fixed by two numbers. $n^*$ itself is stored
exactly in ``breakeven_online_amortized``.
"""


class CostSide(StrEnum):
    """Whose cumulative cost a curve row holds: the compared method's or the reference's."""

    METHOD = "method"
    REFERENCE = "reference"


class BreakEvenState(StrEnum):
    """Whether a compared method's curve meets the reference's.

    ``BREAKS_EVEN``: the method is faster per RHS, so from ``breakeven_online_amortized`` on it
    costs less (0.0 when its setup is no dearer either). ``NEVER``: it is not faster per RHS, so the
    reference stays cheaper for every RHS count. ``NO_COMMON_SYSTEMS``: the pair shares no
    matrix both set up or no system both solved, so there is no mean to draw a curve from.
    """

    BREAKS_EVEN = "breaks_even"
    NEVER = "never"
    NO_COMMON_SYSTEMS = "no_common_systems"


COST_CURVE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "side": pl.String,
    "n_rhs": pl.Int64,
    "matrix_online_amortized_s": pl.Float64,
    "break_even": pl.String,
    "breakeven_online_amortized": pl.Float64,
    "breakeven_matrix_online_amortized_s": pl.Float64,
}
"""One row per compared method, side and RHS count $n = 1..N$. ``side`` holds ``CostSide`` values
and ``break_even`` a ``BreakEvenState``. ``matrix_online_amortized_s`` is the online amortized cost
of one matrix with $n$ RHS, its setup mean plus $n$ solve means, and is null without common
systems. ``breakeven_online_amortized`` ($n^*$, in RHS per matrix) and
``breakeven_matrix_online_amortized_s`` (the cost of either side at $n^*$) are null unless the
method breaks even. The break-even columns repeat on every row of a method."""

_MEANS = ["setup_mean_s", "reference_setup_mean_s", "solve_mean_s", "reference_solve_mean_s"]
_ORDER = ["method", "side", "n_rhs"]


def break_even_curves(obs: pl.DataFrame, reference: PreconditionerName) -> pl.DataFrame:
    r"""Per compared method, the cumulative cost of each side for $n = 1..N$ RHS of one matrix.

    $C(n) = \bar{t}^{\mathrm{setup}} + n\,\bar{t}^{\mathrm{solve},\mathrm{sys}}$, with
    ``COST_CURVE_SCHEMA``. The means are ``break_even``'s, over the population common to the method and the reference,
    so the reference's curve can differ from one compared method to the next. $N$ is the largest
    planned RHS count of a matrix, extended to $\lceil n^* \rceil$ of the furthest break-even
    count so every crossing is on the curves, but not beyond ``BREAK_EVEN_HORIZON_LIMIT``.

    Raises:
        ValueError: The reference is not among the methods.
    """
    means = break_even(obs, reference).with_columns(
        break_even=_state(),
        breakeven_matrix_online_amortized_s=pl.col("setup_mean_s")
        + pl.col("breakeven_online_amortized") * pl.col("solve_mean_s"),
    )
    marks = [
        "method",
        "break_even",
        "breakeven_online_amortized",
        "breakeven_matrix_online_amortized_s",
    ]
    sides = pl.concat(
        [
            means.select(
                *marks,
                side=pl.lit(CostSide.METHOD.value),
                setup=pl.col("setup_mean_s"),
                solve=pl.col("solve_mean_s"),
            ),
            means.select(
                *marks,
                side=pl.lit(CostSide.REFERENCE.value),
                setup=pl.col("reference_setup_mean_s"),
                solve=pl.col("reference_solve_mean_s"),
            ),
        ]
    )
    counts = pl.DataFrame({"n_rhs": range(1, _horizon(obs, means) + 1)}, schema={"n_rhs": pl.Int64})
    return (
        sides.join(counts, how="cross")
        .with_columns(matrix_online_amortized_s=pl.col("setup") + pl.col("n_rhs") * pl.col("solve"))
        .select(list(COST_CURVE_SCHEMA))
        .cast(pl.Schema(COST_CURVE_SCHEMA))
        .sort(_ORDER)
    )


def _state() -> pl.Expr:
    return (
        pl.when(pl.any_horizontal(pl.col(_MEANS).is_null()))
        .then(pl.lit(BreakEvenState.NO_COMMON_SYSTEMS.value))
        .when(pl.col("breakeven_online_amortized").is_null())
        .then(pl.lit(BreakEvenState.NEVER.value))
        .otherwise(pl.lit(BreakEvenState.BREAKS_EVEN.value))
    )


def _horizon(obs: pl.DataFrame, means: pl.DataFrame) -> int:
    """The planned RHS of the largest matrix, or the furthest break-even count up to the limit."""
    planned = int(obs.select(pl.col("planned_rhs").max()).item())
    match means.select(pl.col("breakeven_online_amortized").max()).item():
        case None:
            return planned
        case furthest:
            return max(planned, min(math.ceil(furthest), BREAK_EVEN_HORIZON_LIMIT))
