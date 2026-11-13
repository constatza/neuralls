"""An absolute root that is valid on every OS, for paths tests never touch on disk.

A literal ``/workspace`` is rooted but driveless on Windows, and dlkit's path
resolver rejects such POSIX-rooted paths there. Anchoring at the current
drive gives ``/`` on POSIX and ``C:\\`` (or the active drive) on Windows.
"""

from __future__ import annotations

from pathlib import Path

FAKE_ROOT = Path(Path.cwd().anchor)
