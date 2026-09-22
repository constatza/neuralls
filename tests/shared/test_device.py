"""Tests for ``neuralls.shared.device``."""

from __future__ import annotations

from unittest.mock import patch

from neuralls.shared.device import release_device_memory


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
