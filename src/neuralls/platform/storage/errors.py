"""Error translation shared by every storage backend."""

from __future__ import annotations

from pathlib import Path
from typing import Never


def _raise_storage_error(operation: str, path: Path, exc: OSError) -> Never:
    """Format and raise a descriptive OSError for a storage operation failure.

    Args:
        operation: Human-readable description of the failed operation.
        path: Path involved in the operation.
        exc: The original OSError raised.

    Raises:
        OSError: Always raised with a descriptive message.
    """
    details = [f"{type(exc).__name__}: {exc}"]
    if exc.errno is not None:
        details.append(f"errno={exc.errno}")
    winerror = getattr(exc, "winerror", None)
    if winerror is not None:
        details.append(f"winerror={winerror}")
    if exc.filename:
        details.append(f"src={exc.filename}")
    if exc.filename2:
        details.append(f"dst={exc.filename2}")
    permission_hint = ""
    if isinstance(exc, PermissionError) or winerror == 5:
        permission_hint = (
            " This usually means the filesystem blocked an atomic rename or file lock, "
            "which is common on network shares or when another process is holding the file."
        )
    message = f"{operation} at {path} failed. {', '.join(details)}{permission_hint}"
    raise OSError(message) from exc
