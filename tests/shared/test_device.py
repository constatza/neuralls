"""Tests for ``neuralls.shared.device``."""

from __future__ import annotations

from unittest.mock import patch

import torch

from neuralls.shared.device import release_device_memory, track_resource_usage


def test_release_device_memory_clears_both_device_and_host_caches() -> None:
    """`torch.cuda.empty_cache()` only frees the device caching allocator —
    a `cuMemHostAlloc` OOM comes from the separate pinned-host allocator,
    which needs its own `_host_emptyCache()` call to actually be released.
    """
    with (
        patch("neuralls.shared.device.torch.cuda.is_available", return_value=True),
        patch("neuralls.shared.device.torch.cuda.empty_cache") as empty_cache,
        patch("neuralls.shared.device.torch._C._host_emptyCache") as host_empty_cache,
    ):
        release_device_memory()

    empty_cache.assert_called_once()
    host_empty_cache.assert_called_once()


def test_release_device_memory_noop_without_cuda() -> None:
    with (
        patch("neuralls.shared.device.torch.cuda.is_available", return_value=False),
        patch("neuralls.shared.device.torch.cuda.empty_cache") as empty_cache,
        patch("neuralls.shared.device.torch._C._host_emptyCache") as host_empty_cache,
    ):
        release_device_memory()

    empty_cache.assert_not_called()
    host_empty_cache.assert_not_called()


def test_track_resource_usage_reports_real_memory_on_cpu() -> None:
    """CPU memory is a real, non-None number, not silently skipped — an
    ``rusage`` delta, so it must be a non-negative int rather than `None`.
    """
    with track_resource_usage(torch.device("cpu")) as usage:
        _ = torch.zeros(1000, 1000)

    result = usage()
    assert result.wall_time_seconds >= 0
    assert isinstance(result.peak_memory_bytes, int)
    assert result.peak_memory_bytes >= 0


def test_track_resource_usage_reads_cuda_peak_when_available() -> None:
    """On CUDA, memory comes from the exact scoped `torch.cuda` peak-memory API."""
    with (
        patch("neuralls.shared.device.torch.cuda.synchronize") as sync,
        patch("neuralls.shared.device.torch.cuda.reset_peak_memory_stats") as reset_peak,
        patch(
            "neuralls.shared.device.torch.cuda.max_memory_allocated", return_value=12345
        ) as max_allocated,
    ):
        with track_resource_usage(torch.device("cuda")) as usage:
            pass
        result = usage()

    assert reset_peak.call_count == 1
    assert sync.call_count == 2
    max_allocated.assert_called_once()
    assert result.peak_memory_bytes == 12345
    assert result.wall_time_seconds >= 0
