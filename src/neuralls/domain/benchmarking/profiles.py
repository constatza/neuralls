"""Profiles of how many systems each preconditioner solves, as series over the observation view.

Both profiles are cumulative distributions over systems in which a failed run
stays an unsolved system. Nothing is averaged and no penalty is invented:

- Performance profile (Dolan & Moré 2002, "Benchmarking optimization software
  with performance profiles", Math. Program. 91): the fraction of systems on
  which a method's cost is within a factor tau of the best method's cost on
  that system. A failure is ratio infinity.
- ECDF profile (Hansen et al. 2016, "COCO: Performance Assessment",
  arXiv:1605.03560): the fraction of systems solved within an absolute budget
  of iterations or seconds. It does not depend on which other methods are
  compared. (Moré & Wild's 2009 data profile additionally divides the budget by
  n_p + 1; this one does not.)

Gould & Scott (2016, ACM TOMS 43) caution that a profile is not an estimate of
expected performance, so these are shown beside explicit success counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from neuralls.domain.solver.reporting.metrics import MetricId, system_metric
from neuralls.domain.solver.reporting.summary import Population, select

PROFILE_SERIES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "x": pl.Float64,
    "fraction": pl.Float64,
    "n_planned": pl.Int64,
    "n_solved": pl.Int64,
}
"""One row per system a method solved, ascending in ``x``; ``fraction`` is solved-within-``x`` over
``n_planned``, so a curve ends at ``n_solved / n_planned``."""

PLANNED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "n_planned": pl.Int64,
    "n_solved": pl.Int64,
}
"""One row per method, including a method that solved nothing and so has no curve row."""

_SYSTEM_KEYS = ["dataset", "system_id"]


@dataclass(frozen=True, slots=True, eq=False)
class ProfileSeries:
    """The curves of a profile as frames, with the counts that place their plateaus.

    Attributes:
        frame: ``PROFILE_SERIES_SCHEMA`` rows of the solved systems of every method.
        planned: ``PLANNED_SCHEMA`` rows, one per method of the observations. A method that
            solved nothing has no row in ``frame`` and is still here with ``n_solved = 0``.
        n_excluded_zero_best: Systems left out of every method's counts because the best method
            needed a value of 0 on them (always 0 for an absolute ECDF).
    """

    frame: pl.DataFrame
    planned: pl.DataFrame
    n_excluded_zero_best: int


def ecdf_series(obs: pl.DataFrame, metric: MetricId) -> ProfileSeries:
    """Per method, the fraction of its planned systems solved within each absolute budget.

    The absolute-budget ECDF of ``metric`` (Hansen et al. 2016): ``x`` is the metric value of a
    system the method solved, as ``select`` defines it, and ``fraction`` divides by every planned
    system of the method, so unsolved systems keep the curve below 1. It does not depend on which
    other methods are compared.

    Raises:
        ValueError: The metric has one value per matrix.
    """
    spec = system_metric(metric)
    solved = select(obs, metric, Population.OWN)
    points = solved.select("method", x=pl.col(spec.column).cast(pl.Float64))
    return _series(obs, obs, points, n_excluded_zero_best=0)


def performance_series(obs: pl.DataFrame, metric: MetricId) -> ProfileSeries:
    r"""Per method, the fraction of planned systems within each factor of the best method.

    The Dolan-More performance profile of ``metric``: on a system the ratio of a solving method
    is ``value / best`` with ``best`` the least value over the methods that solved it, and
    $\rho_{p,s} = x_{p,s} / \min_{p'} x_{p',s}$. A system a method did not solve has no ratio
    but stays in its denominator, so the curve plateaus below 1.

    A system whose best value is 0 (an iteration count of a solve at the initial guess, or a time
    of zero) is excluded from numerator and denominator of every method and counted in
    ``n_excluded_zero_best``. Its ratios are undefined, and keeping it in the denominators would
    lower every method's plateau by a system nobody failed on, which reads as failures.

    Raises:
        ValueError: The metric has one value per matrix.
    """
    spec = system_metric(metric)
    solved = select(obs, metric, Population.OWN)
    value = pl.col(spec.column).cast(pl.Float64)
    best = solved.group_by(_SYSTEM_KEYS).agg(best=value.min())
    zero_best = best.filter(pl.col("best") == 0.0).select(_SYSTEM_KEYS)
    points = solved.join(best.filter(pl.col("best") != 0.0), on=_SYSTEM_KEYS, how="inner").select(
        "method", x=value / pl.col("best")
    )
    planned = obs.join(zero_best, on=_SYSTEM_KEYS, how="anti")
    return _series(obs, planned, points, n_excluded_zero_best=zero_best.height)


def _series(
    obs: pl.DataFrame, planned: pl.DataFrame, points: pl.DataFrame, *, n_excluded_zero_best: int
) -> ProfileSeries:
    """Assemble the curves from the solved ``points`` and the systems each method planned."""
    counts = (
        obs.select("method")
        .unique()
        .join(planned.group_by("method").agg(n_planned=pl.len()), on="method", how="left")
        .join(points.group_by("method").agg(n_solved=pl.len()), on="method", how="left")
        .with_columns(pl.col("n_planned", "n_solved").fill_null(0))
        .select(list(PLANNED_SCHEMA))
        .cast(pl.Schema(PLANNED_SCHEMA))
        .sort("method")
    )
    frame = (
        points.with_columns(solved_within=pl.col("x").rank("max").over("method"))
        .join(counts, on="method", how="left")
        .with_columns(fraction=pl.col("solved_within") / pl.col("n_planned"))
        .select(list(PROFILE_SERIES_SCHEMA))
        .cast(pl.Schema(PROFILE_SERIES_SCHEMA))
        .sort("method", "x")
    )
    return ProfileSeries(frame, counts, n_excluded_zero_best)
