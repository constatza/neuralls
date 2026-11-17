"""How many times a timed region is measured, and how repeats collapse to one number."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from neuralls.shared.constants import MAX_TIMING_REPEATS


class TimingStatistic(StrEnum):
    """How repeated wall-time samples are reduced to the single reported number."""

    MEDIAN = "median"
    MEAN = "mean"
    MIN = "min"
    """Best-of-k: the least noise-contaminated sample, as `timeit` recommends."""

    def reduce(self, samples: Sequence[float]) -> float:
        """Collapse non-empty `samples` to one value."""
        return float(self.reduce_along(np.asarray(samples, dtype=np.float64), axis=0))

    def reduce_along(self, samples: np.ndarray, axis: int) -> np.ndarray:
        """Collapse `samples` along `axis`, the same reduction `reduce` applies to one event."""
        match self:
            case TimingStatistic.MEDIAN:
                return np.median(samples, axis=axis)
            case TimingStatistic.MEAN:
                return np.mean(samples, axis=axis)
            case TimingStatistic.MIN:
                return np.min(samples, axis=axis)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class TimingRepeatPolicy:
    """Measurement effort for one timed region (a solve, or a preconditioner build).

    The default is one discarded warm-up, then three timed executions reduced by their
    median. The warm-up absorbs first-call costs that `shared.device.warm_up_device` does
    not pre-pay, because they belong to kernels only a real build or solve reaches: the
    LU, SVD and eigensolver kernels and the damping (omega) estimate of a setup. Three
    samples give a median, the reduction Hoefler & Belli recommend (SC'15, "Scientific
    benchmarking of parallel computing systems"): run times are right-skewed by
    interference, and the median is robust to it where the mean is not. A case that needs
    cheap runs sets `repeats = 1` and `warmup = 0`; every execution repeats the whole region
    for every (system, preconditioner), so the effort is capped by `MAX_TIMING_REPEATS`.

    Attributes:
        repeats: Timed executions whose samples are reduced by `statistic`.
        warmup: Untimed executions run first and discarded (neither their time nor their
            peak memory is stored), absorbing one-off costs (lazy imports, allocator
            growth, cold caches, first calls of factorization kernels). Costs that occur
            once per process (CUDA context, cuBLAS/cuSPARSE handles, kernel loading) are
            instead paid once by `shared.device.warm_up_device` before the first
            measurement, and checkpoint loading is kept outside the timed region
            altogether (`PreconditionerService.preload`).
        statistic: Reduction applied to the timed samples.
    """

    repeats: Annotated[int, Field(ge=1, le=MAX_TIMING_REPEATS)] = 3
    warmup: Annotated[int, Field(ge=0, le=MAX_TIMING_REPEATS)] = 1
    statistic: TimingStatistic = TimingStatistic.MEDIAN
