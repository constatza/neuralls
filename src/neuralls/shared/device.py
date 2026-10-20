"""Cross-layer CUDA device-memory primitives."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Generator, Hashable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from loguru import logger

from neuralls.shared.timing import TimingRepeatPolicy

if sys.platform == "win32":
    import psutil
else:
    import resource


def _sample_peak_rss_kib() -> int:
    """Process-wide peak RSS in KiB, POSIX `ru_maxrss` units on every platform.

    POSIX `rusage` has no reset API, so `ru_maxrss` is a monotonic
    high-water mark for the process's whole lifetime — the same is true of
    Windows' `peak_wset`, so both give the same "under-reports if an earlier
    region already hit a bigger peak" behavior documented on
    `end_resource_usage`, just via different APIs (`resource` isn't present
    on Windows at all — see `ModuleNotFoundError: No module named 'resource'`).
    """
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset // 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Wall time and peak memory for one measured region.

    Attributes:
        wall_time_seconds: Elapsed wall-clock time for the region.
        peak_memory_bytes: On CUDA, an exact scoped peak (the allocator's
            high-water mark reset at region entry). On CPU, a peak-RSS
            delta — see `track_resource_usage`.
    """

    wall_time_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceUsageToken:
    """State carried from `begin_resource_usage` to `end_resource_usage`.

    Public (not `track_resource_usage`-internal) so callers who can't use a
    single `with` block — e.g. lifecycle hooks bracketing an
    externally-orchestrated sweep child — can hold it across two separate
    callback invocations.
    """

    device: torch.device
    start: float
    rss_before: int | None


def begin_resource_usage(device: torch.device) -> ResourceUsageToken:
    """Start a resource-usage measurement, GPU-synchronized, for later `end_resource_usage`.

    Exists alongside `track_resource_usage` for callers that can't use a
    single `with` block because the start and end of the measured region
    happen in two separate callback invocations (e.g. lifecycle hooks
    bracketing an externally-orchestrated multirun sweep child) —
    `track_resource_usage` itself is a thin wrapper over this pair.

    Args:
        device: Device the measured region runs on.

    Returns:
        Token to pass to `end_resource_usage`.
    """
    match device.type:
        case "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            rss_before = None
        case _:
            rss_before = _sample_peak_rss_kib()
    return ResourceUsageToken(device=device, start=time.perf_counter(), rss_before=rss_before)


def end_resource_usage(token: ResourceUsageToken) -> ResourceUsage:
    """Finish a measurement started by `begin_resource_usage`, GPU-synchronized.

    CUDA kernels launch asynchronously — without `torch.cuda.synchronize()`
    immediately before starting the clock and immediately before stopping it,
    the measured wall time would reflect host-side kernel-launch overhead, not
    the device work actually being timed.

    Memory is always a real number on both CPU and GPU, never `None` — a
    GPU-only reading would leave every CPU-run comparison with zero memory
    visibility, which is a worse signal than an approximate one:
    - CUDA: `torch.cuda.max_memory_allocated` after
      `torch.cuda.reset_peak_memory_stats` (done in `begin_resource_usage`),
      an exact peak scoped to this call.
    - CPU: peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss` on POSIX,
      `psutil.Process().memory_info().peak_wset` on Windows) sampled before
      and after, delta reported (0 if no new peak was set). Both are
      process-wide, monotonic high-water marks with no reset API, so this
      under-reports if the process already hit a bigger peak earlier (e.g. a
      prior preconditioner's build in the same run).
      # ponytail: process-wide high-water mark, not a true scoped peak; swap
      # for a sampling thread or psutil if a precise per-call CPU peak is
      # ever needed.

    Args:
        token: Token returned by `begin_resource_usage`.

    Returns:
        The measured `ResourceUsage`.
    """
    device = token.device
    match device.type:
        case "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        case _:
            rss_after = _sample_peak_rss_kib()
            peak_memory_bytes = max(rss_after - (token.rss_before or 0), 0) * 1024
    return ResourceUsage(
        wall_time_seconds=time.perf_counter() - token.start,
        peak_memory_bytes=peak_memory_bytes,
    )


@contextmanager
def track_resource_usage(device: torch.device) -> Generator[Callable[[], ResourceUsage]]:
    """Time a block and read its peak memory, via `begin_resource_usage`/`end_resource_usage`.

    Args:
        device: Device the measured region runs on.

    Yields:
        A zero-arg accessor for the `ResourceUsage`, valid only after the
        `with` block exits (measuring end-of-region state requires the block
        to have finished).

    Example:
        >>> with track_resource_usage(torch.device("cpu")) as usage:
        ...     _ = torch.zeros(10)
        >>> usage().wall_time_seconds >= 0
        True
    """
    token = begin_resource_usage(device)
    box: list[ResourceUsage] = []
    try:
        yield lambda: box[0]
    finally:
        box.append(end_resource_usage(token))


@dataclass(frozen=True, slots=True)
class TimedRuns[T]:
    """Outcome of `repeat_timed`.

    Attributes:
        value: What the first timed execution returned.
        seconds: The policy's statistic over `samples_seconds`.
        samples_seconds: Wall time of every timed execution, in run order —
            kept so the spread behind `seconds` stays visible.
        peak_memory_bytes: Peak memory of the very first execution (warm-up
            included). CPU peak RSS is a process-wide high-water mark with no
            reset, so any later execution would report ~0.
    """

    value: T
    seconds: float
    samples_seconds: tuple[float, ...]
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class FailedTimedRuns:
    """A timed execution that raised after still producing resource data."""

    error: Exception
    seconds: float
    samples_seconds: tuple[float, ...]
    peak_memory_bytes: int


type TimedOutcome[T] = TimedRuns[T] | FailedTimedRuns


@dataclass(slots=True)
class WorkloadMemoryObserver:
    """Observe a method lifetime without resetting per-operation CUDA peaks.

    CUDA operation timers keep ownership of ``reset_peak_memory_stats``.  The
    observer combines their exact peaks with allocator snapshots between
    operations, avoiding nested resets.  POSIX ``ru_maxrss`` (and Windows
    ``peak_wset``) is process-wide and monotonic, so the CPU result is only a
    lower bound when an earlier workload established a higher process peak.
    """

    device: torch.device
    _baseline_rss_kib: int | None
    _peak_memory_bytes: int

    @classmethod
    def start(cls, device: torch.device) -> WorkloadMemoryObserver:
        """Begin observation without changing allocator counters."""
        if device.type == "cuda":
            return cls(device, None, torch.cuda.memory_allocated(device))
        return cls(device, _sample_peak_rss_kib(), 0)

    def observe_operation(self, peak_memory_bytes: int) -> None:
        """Include one independently measured setup or solve peak."""
        self._peak_memory_bytes = max(self._peak_memory_bytes, peak_memory_bytes)
        if self.device.type == "cuda":
            self._peak_memory_bytes = max(
                self._peak_memory_bytes, torch.cuda.memory_allocated(self.device)
            )

    def finish(self) -> int:
        """Return the best available method-lifetime peak in bytes."""
        if self.device.type == "cuda":
            return max(self._peak_memory_bytes, torch.cuda.memory_allocated(self.device))
        current = _sample_peak_rss_kib()
        process_delta = max(current - (self._baseline_rss_kib or 0), 0) * 1024
        return max(self._peak_memory_bytes, process_delta)


def _measure[T](run: Callable[[], T], device: torch.device) -> tuple[T, ResourceUsage]:
    """One GPU-synchronised execution of `run` and its resource usage."""
    with track_resource_usage(device) as usage:
        value = run()
    return value, usage()


def _measure_outcome[T](
    run: Callable[[], T], device: torch.device
) -> tuple[T | Exception, ResourceUsage]:
    """Measure one execution while retaining resource data on ordinary failure."""
    result: T | Exception = RuntimeError("timed execution did not run")
    with track_resource_usage(device) as usage:
        try:
            result = run()
        except Exception as exc:  # noqa: BLE001 - the caller models failure as data
            result = exc
    return result, usage()


def repeat_timed_outcome[T](
    run: Callable[[], T],
    policy: TimingRepeatPolicy,
    *,
    device: torch.device,
    fingerprint: Callable[[T], Hashable] | None = None,
) -> TimedOutcome[T]:
    """Repeat ``run`` like :func:`repeat_timed`, preserving a timed failure."""
    for _ in range(policy.warmup):
        warmup_value, warmup_usage = _measure_outcome(run, device)
        if isinstance(warmup_value, Exception):
            return FailedTimedRuns(
                warmup_value,
                warmup_usage.wall_time_seconds,
                (warmup_usage.wall_time_seconds,),
                warmup_usage.peak_memory_bytes,
            )

    first_value, first_usage = _measure_outcome(run, device)
    samples = [first_usage.wall_time_seconds]
    if isinstance(first_value, Exception):
        return FailedTimedRuns(
            first_value,
            policy.statistic.reduce(samples),
            tuple(samples),
            first_usage.peak_memory_bytes,
        )
    reference = fingerprint(first_value) if fingerprint is not None else None
    for repeat in range(1, policy.repeats):
        value, usage = _measure_outcome(run, device)
        samples.append(usage.wall_time_seconds)
        if isinstance(value, Exception):
            return FailedTimedRuns(
                value,
                policy.statistic.reduce(samples),
                tuple(samples),
                max(first_usage.peak_memory_bytes, usage.peak_memory_bytes),
            )
        if fingerprint is not None and fingerprint(value) != reference:
            logger.warning(
                "Timed repeat {} differs from the first ({} != {}); its timing is not comparable.",
                repeat + 1,
                fingerprint(value),
                reference,
            )
    return TimedRuns(
        value=first_value,
        seconds=policy.statistic.reduce(samples),
        samples_seconds=tuple(samples),
        peak_memory_bytes=first_usage.peak_memory_bytes,
    )


def repeat_timed[T](
    run: Callable[[], T],
    policy: TimingRepeatPolicy,
    *,
    device: torch.device,
    fingerprint: Callable[[T], Hashable] | None = None,
) -> TimedRuns[T]:
    """Execute `run` per `policy` and reduce its wall times to one number.

    Runs `policy.warmup` untimed executions, then `policy.repeats` timed ones,
    each GPU-synchronised through `track_resource_usage`. The default policy is
    exactly one timed execution, i.e. a plain measurement.

    Args:
        run: The region to measure. Must be repeatable (fresh state each call).
        policy: Warm-up count, repeat count and reduction.
        device: Device the region runs on.
        fingerprint: Optional summary of each result (e.g. an iteration count).
            A repeat whose summary differs from the first logs a warning,
            since a nondeterministic or stateful region makes its samples
            incomparable.

    Returns:
        The first timed value, the reduced time, every sample, and the first
        execution's peak memory.
    """
    warmup_usages = [_measure(run, device)[1] for _ in range(policy.warmup)]
    first_value, first_usage = _measure(run, device)
    samples = [first_usage.wall_time_seconds]
    reference = fingerprint(first_value) if fingerprint is not None else None
    for repeat in range(1, policy.repeats):
        value, usage = _measure(run, device)
        samples.append(usage.wall_time_seconds)
        if fingerprint is not None and fingerprint(value) != reference:
            logger.warning(
                "Timed repeat {} differs from the first ({} != {}); its timing is not comparable.",
                repeat + 1,
                fingerprint(value),
                reference,
            )
    first_execution = warmup_usages[0] if warmup_usages else first_usage
    return TimedRuns(
        value=first_value,
        seconds=policy.statistic.reduce(samples),
        samples_seconds=tuple(samples),
        peak_memory_bytes=first_execution.peak_memory_bytes,
    )


def release_device_memory() -> None:
    """Return cached CUDA device *and* pinned-host blocks to the driver (no-op without CUDA).

    Shared by every layer that runs a sequence of CUDA-heavy operations in
    one long-lived process without an intervening context reset: solver
    comparisons (between preconditioners), the training sweep (between
    fit-job children), and the case pipeline (between the training and
    comparison stages) — none of these hand memory back on their own, so a
    failed or oversized allocation in one leaves the caching allocator
    holding blocks the next one needs.

    `torch.cuda.empty_cache()` only frees the *device* caching allocator
    (what `nvidia-smi` shows) — it does nothing for the separate pinned
    (page-locked) host-memory caching allocator that `cudaHostAlloc`/
    `cuMemHostAlloc` draws from. A `CUDA_ERROR_OUT_OF_MEMORY` raised from
    `cuMemHostAlloc` is host-RAM exhaustion, not VRAM exhaustion, and
    `empty_cache()` alone leaves it unaddressed — hence `_host_emptyCache()`
    (torch's private counterpart for the host allocator; no public wrapper
    exists yet, same as `empty_cache()` wrapping `_cuda_emptyCache()`).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()


WARMUP_SYSTEM_SIZE = 64
"""Rows of the throwaway SPD system `warm_up_device` runs on: big enough to hit real kernels."""

WARMUP_CG_ITERATIONS = 5
"""CG-style iterations `warm_up_device` runs, enough to touch every op CG's loop uses."""


def warm_up_device(device: torch.device) -> None:
    """Pay a device's first-call costs once, outside every timed region.

    The first use of dense and sparse kernels in a process carries one-off costs
    unrelated to any algorithm: CUDA context and cuBLAS/cuSPARSE handle creation,
    kernel loading, allocator growth, lazy imports. Without this, whichever
    preconditioner is measured first absorbs them and the comparison depends on
    configuration order. This runs a tiny throwaway workload covering the ops the
    preconditioners and CG use (dense matmul and triangular solve, CSR matvec, and
    the dot/axpy loop of CG) on a small SPD system.

    This is a per-*process* warm-up. It is deliberately not the per-run
    `TimingRepeatPolicy.warmup`, which repeats a whole timed build or solve for
    every (system, preconditioner) just to absorb costs that occur once.

    Args:
        device: Solver device to warm.
    """
    n = WARMUP_SYSTEM_SIZE
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(n, n, generator=generator, dtype=torch.float64)
    matrix = (base @ base.T + n * torch.eye(n, dtype=torch.float64)).to(device)
    rhs = torch.ones(n, dtype=torch.float64, device=device)
    csr = matrix.to_sparse_csr()

    torch.linalg.solve_triangular(torch.tril(matrix), rhs[:, None], upper=False)
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    for _ in range(WARMUP_CG_ITERATIONS):
        for operator in (matrix, csr):
            applied = operator @ direction
        alpha = (residual @ residual) / (direction @ applied)
        x = x + alpha * direction
        new_residual = residual - alpha * applied
        direction = (
            new_residual + ((new_residual @ new_residual) / (residual @ residual)) * direction
        )
        residual = new_residual
    if device.type == "cuda":
        torch.cuda.synchronize(device)
