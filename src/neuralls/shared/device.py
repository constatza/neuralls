"""Cross-layer CUDA device-memory primitives."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

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
def track_resource_usage(device: torch.device) -> Iterator[Callable[[], ResourceUsage]]:
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
