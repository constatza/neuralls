r"""The average residual reduction per iteration of each system, from its stored history.

A stored ``convergence`` frame holds, per method and system, the relative residual
$\hat{r}_{p,s,j} = \|r_j\| / \|b\|$ at iterations $j = 0..J_{p,s}$, $j = 0$ being the initial
residual. The convergence factor is the geometric mean of the per-iteration reductions,

$$\gamma_{p,s} = \left(\frac{\hat{r}_{p,s,J}}{\hat{r}_{p,s,0}}\right)^{1/J_{p,s}},$$

so it does not depend on how the reduction was spread over the iterations. It is undefined (null)
without an iteration ($J = 0$) or for a zero initial residual (a zero right-hand side), where no
reduction exists; a zero final residual gives 0. Which outcomes it is reported on is the metric
catalog's rule, not this module's.
"""

from __future__ import annotations

import polars as pl

CONVERGENCE_FACTOR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "system_id": pl.String,
    "convergence_factor": pl.Float64,
}
"""One row per method and system with a stored history."""

_KEYS = ["method", "system_id"]


def convergence_factors(convergence: pl.DataFrame) -> pl.DataFrame:
    """The convergence factor of every system of a ``CONVERGENCE_SCHEMA`` frame.

    The history's first and last stored iterations bound it, so a frame stored in any row order
    gives the same factor.
    """
    iteration = pl.col("iteration")
    residual = pl.col("residual_rel")
    per_system = convergence.group_by(_KEYS).agg(
        steps=(iteration.max() - iteration.min()).cast(pl.Float64),
        initial=residual.sort_by(iteration).first(),
        final=residual.sort_by(iteration).last(),
    )
    defined = (pl.col("steps") >= 1) & (pl.col("initial") > 0.0)
    return (
        per_system.with_columns(
            convergence_factor=pl.when(defined).then(
                (pl.col("final") / pl.col("initial")).pow(1.0 / pl.col("steps"))
            )
        )
        .select(list(CONVERGENCE_FACTOR_SCHEMA))
        .cast(pl.Schema(CONVERGENCE_FACTOR_SCHEMA))
        .sort(_KEYS)
    )
