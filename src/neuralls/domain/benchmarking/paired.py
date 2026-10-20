"""Failure-aware paired ratios with matrix-clustered uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from neuralls.domain.benchmarking.costs import amortized_online_seconds
from neuralls.domain.benchmarking.uncertainty import matrix_cluster_median_interval
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.domain.solver.models.statistics import PairedComparison
from neuralls.shared.constants import MIN_PAIRED_SYSTEMS
from neuralls.shared.types import MatrixId, SystemId


@dataclass(frozen=True, slots=True)
class PairedPoints:
    """Solved observations split by whether both, only method, or only reference solved."""

    both: tuple[tuple[Solved, Solved], ...]
    method_only: tuple[Solved, ...]
    reference_only: tuple[Solved, ...]


def _positive_time(outcome: SystemOutcome | None) -> float | None:
    if not isinstance(outcome, Solved):
        return None
    value = amortized_online_seconds(outcome)
    return value if value > 0 else None


def paired_points(
    reference: Mapping[SystemId, SystemOutcome], method: Mapping[SystemId, SystemOutcome]
) -> PairedPoints:
    """Split common systems without imputing a numeric cost for failures."""
    both: list[tuple[Solved, Solved]] = []
    method_only: list[Solved] = []
    reference_only: list[Solved] = []
    for system_id, ref_outcome in reference.items():
        method_outcome = method.get(system_id)
        if method_outcome is None:
            continue
        if isinstance(ref_outcome, Solved) and isinstance(method_outcome, Solved):
            both.append((ref_outcome, method_outcome))
        elif isinstance(ref_outcome, Failed) and isinstance(method_outcome, Solved):
            method_only.append(method_outcome)
        elif isinstance(ref_outcome, Solved) and isinstance(method_outcome, Failed):
            reference_only.append(ref_outcome)
    return PairedPoints(tuple(both), tuple(method_only), tuple(reference_only))


def _wilcoxon_p(log_ratios: np.ndarray) -> float | None:
    """Two-sided signed-rank p-value of log(reference / method) ratios."""
    if np.count_nonzero(log_ratios) < MIN_PAIRED_SYSTEMS:
        return None
    return float(wilcoxon(log_ratios).pvalue)


def compare_paired(points: PairedPoints, *, seed: int) -> PairedComparison:
    """Summarize paired ratios and asymmetric successes without failure penalties."""
    positive_pairs = tuple(
        (reference, method)
        for reference, method in points.both
        if _positive_time(reference) is not None and _positive_time(method) is not None
    )
    ratios = tuple(
        amortized_online_seconds(reference) / amortized_online_seconds(method)
        for reference, method in positive_pairs
    )
    ratios_by_matrix: dict[MatrixId, list[float]] = {}
    for (reference, _), ratio in zip(positive_pairs, ratios, strict=True):
        ratios_by_matrix.setdefault(reference.matrix_id, []).append(ratio)
    values = np.asarray(ratios, dtype=np.float64)
    return PairedComparison(
        speed_ratios=ratios,
        median_speed_ratio=float(np.median(values)) if ratios else None,
        median_speed_ratio_ci=matrix_cluster_median_interval(ratios_by_matrix, seed=seed),
        n_both_solved=len(points.both),
        n_reference_only=len(points.reference_only),
        n_method_only=len(points.method_only),
        wilcoxon_p=_wilcoxon_p(np.log(values)) if ratios else None,
    )
