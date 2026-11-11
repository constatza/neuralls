"""Tests for ``neuralls.shared.device``."""

from __future__ import annotations

from unittest.mock import patch

import torch

from neuralls.shared.device import track_resource_usage


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
