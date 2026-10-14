"""How many times a timed region is measured, and how repeats collapse to one number."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

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
        match self:
            case TimingStatistic.MEDIAN:
                return statistics.median(samples)
            case TimingStatistic.MEAN:
                return statistics.fmean(samples)
            case TimingStatistic.MIN:
                return min(samples)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class TimingRepeatPolicy:
    """Measurement effort for one timed region (a solve, or a preconditioner build).

    The default (one timed run, no warm-up) is the cheapest possible and is what
    a case gets unless it asks for better estimates; every system can be
    expensive, so the effort is opt-in and capped by `MAX_TIMING_REPEATS`.

    Attributes:
        repeats: Timed executions whose samples are reduced by `statistic`.
        warmup: Untimed executions run first, absorbing one-off costs (lazy
            imports, allocator growth, cold caches). Each one repeats the whole
            timed region (a full build or solve) for every (system,
            preconditioner), so it is opt-in. Costs that occur once per process
            (CUDA context, cuBLAS/cuSPARSE handles, kernel loading) are instead
            paid once by `shared.device.warm_up_device` before the first
            measurement, and checkpoint loading is kept outside the timed region
            altogether (`PreconditionerService.preload`).
        statistic: Reduction applied to the timed samples.
    """

    repeats: Annotated[int, Field(ge=1, le=MAX_TIMING_REPEATS)] = 1
    warmup: Annotated[int, Field(ge=0, le=MAX_TIMING_REPEATS)] = 0
    statistic: TimingStatistic = TimingStatistic.MEDIAN
