"""Hashing utilities for workflow caching and change detection."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def compute_dataset_fingerprint(paths: Iterable[Path]) -> str:
    """Compute a stable fingerprint of dataset artifact files from size + mtime.

    Cheap alternative to hashing file contents: changes whenever a dataset is
    regenerated (its output files' size/mtime differ), without reading
    potentially large array data. Used to invalidate a cached "training
    already completed" result when the underlying dataset has been
    regenerated since that training run.

    Args:
        paths: Dataset artifact file paths to fingerprint (order-independent).

    Returns:
        SHA-1 hash of each path's size and modification time.
    """
    hasher = hashlib.sha1()
    for path in sorted(paths):
        stat = path.stat()
        hasher.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()
