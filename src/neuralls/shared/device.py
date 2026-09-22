"""Cross-layer CUDA device-memory primitives."""

from __future__ import annotations

import torch


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
