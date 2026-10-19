"""Paired comparison of a method against a reference on the same systems.

Only systems both methods solved give a speedup; a system one method failed
and the other solved is kept apart (never given a penalty time) so it can be
reported as a robustness difference. References: Wilcoxon signed-rank test on
paired differences (Wilcoxon 1945, "Individual comparisons by ranking
methods", Biometrics Bulletin 1) and the percentile bootstrap of the median
(Efron & Tibshirani 1993, "An Introduction to the Bootstrap").

Times are the recurring time per system (`profiles.seconds`): one-time shared
costs are excluded here and charged once in ERT, the projection and break-even.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from neuralls.domain.benchmarking.costs import seconds
from neuralls.domain.benchmarking.uncertainty import median_interval
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.domain.solver.models.statistics import PairedComparison
from neuralls.shared.constants import MIN_PAIRED_SYSTEMS


@dataclass(frozen=True, slots=True)
class PairedPoints:
    """Times of a method and a reference, split by which of them solved a system.

    Attributes:
        both: (reference time, method time) for systems both solved.
        method_only: Method times on systems the reference failed.
        reference_only: Reference times on systems the method failed.
    """

    both: tuple[tuple[float, float], ...]
    method_only: tuple[float, ...]
    reference_only: tuple[float, ...]


def _solved_time(outcome: SystemOutcome | None) -> float | None:
    """The comparable time of a solved run, `None` otherwise."""
    return seconds(outcome) if isinstance(outcome, Solved) else None


def paired_points(
    reference: Mapping[str, SystemOutcome], method: Mapping[str, SystemOutcome]
) -> PairedPoints:
    """Split the systems both methods ran into paired, method-only and reference-only times.

    Systems with an incomparable time, a double failure, or run by only one
    method are skipped.
    """
    both: list[tuple[float, float]] = []
    method_only: list[float] = []
    reference_only: list[float] = []
    for system_id, ref_outcome in reference.items():
        if system_id not in method:
            continue
        method_outcome = method[system_id]
        ref_time = _solved_time(ref_outcome)
        method_time = _solved_time(method_outcome)
        if ref_time is not None and method_time is not None:
            both.append((ref_time, method_time))
        elif method_time is not None and isinstance(ref_outcome, Failed):
            method_only.append(method_time)
        elif ref_time is not None and isinstance(method_outcome, Failed):
            reference_only.append(ref_time)
    return PairedPoints(tuple(both), tuple(method_only), tuple(reference_only))


def _wilcoxon_p(log_speedups: np.ndarray) -> float | None:
    """Two-sided signed-rank p-value of the log speedups against zero.

    Testing log ratios is the same test as on log times and agrees with the
    ratio-valued median speedup (times are multiplicative and heavy-tailed).
    `None` with fewer than `MIN_PAIRED_SYSTEMS` non-zero differences, since
    zeros are dropped and the remaining pairs carry the test.
    """
    if np.count_nonzero(log_speedups) < MIN_PAIRED_SYSTEMS:
        return None
    return float(wilcoxon(log_speedups).pvalue)


def compare_paired(points: PairedPoints, *, seed: int) -> PairedComparison | None:
    """Speedups (reference time / method time) on the systems both solved, or `None` without pairs.

    The median speedup carries a seeded bootstrap interval and the time
    differences a Wilcoxon signed-rank p-value when enough pairs exist.
    """
    if not points.both:
        return None
    pairs = np.asarray(points.both, dtype=np.float64)
    speedups = pairs[:, 0] / pairs[:, 1]
    return PairedComparison(
        n_both_solved=len(points.both),
        speedups=tuple(float(s) for s in speedups),
        median_speedup=float(np.median(speedups)),
        median_speedup_ci=median_interval(tuple(float(s) for s in speedups), seed=seed),
        wilcoxon_p=_wilcoxon_p(np.log(speedups)),
    )
