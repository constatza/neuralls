"""Atomic commit of a generated dataset directory.

A dataset is written into a sibling staging directory, ``<name>.partial``, and renamed to
its final name only after every array and the manifest are written. The final name
therefore either does not exist or holds a complete dataset; an interrupted run leaves
only the staging directory, which the next run removes before it starts.

Replacing an existing final directory (a forced regeneration) moves the old dataset aside
to ``<name>.old`` before the staged one is renamed into place, so a failed swap can put the
old dataset back. A leftover ``.old`` from a crash is restored to the final name when the
final name is missing, and deleted when the final name exists. Stale ``.partial``
directories are removed before a new commit starts.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from neuralls.platform.storage.errors import _raise_storage_error

PARTIAL_SUFFIX = ".partial"
"""Suffix of the staging directory that sits next to the final dataset directory."""

ASIDE_SUFFIX = ".old"
"""Suffix of the directory that holds the previous dataset while a replacement is committed."""

RENAME_ATTEMPTS = 5
"""Total tries for one rename before a persistent PermissionError is reported."""

RENAME_RETRY_DELAY_SECONDS = 0.1
"""Pause between rename tries, long enough for antivirus or indexing to release a file."""

type StagedWrite = Callable[[Path], bool]
"""Writes a dataset into the staging directory. Returns False when nothing was written."""


def aside_dir_for(final_dir: Path) -> Path:
    """Return the directory that holds the previous dataset during a replacement.

    Args:
        final_dir: Final dataset directory.

    Returns:
        ``<final_dir name>.old`` in the same parent directory.
    """
    return final_dir.with_name(final_dir.name + ASIDE_SUFFIX)


def staging_dir_for(final_dir: Path) -> Path:
    """Return the staging directory for ``final_dir``, on the same filesystem.

    Args:
        final_dir: Final dataset directory.

    Returns:
        ``<final_dir name>.partial`` in the same parent directory.
    """
    return final_dir.with_name(final_dir.name + PARTIAL_SUFFIX)


def commit_staged_directory(final_dir: Path, write: StagedWrite) -> bool:
    """Write a dataset into staging and rename it to ``final_dir`` on success.

    Stale staging and aside directories from an earlier interrupted run are removed first. If ``write``
    raises, the staging directory is left in place and the error propagates, so the final
    name is never created. If ``write`` returns False, the staging directory is removed
    and nothing is committed.

    Args:
        final_dir: Final dataset directory to create.
        write: Callable that writes every artifact and the manifest into the directory it
            receives, and returns True when it wrote a dataset.

    Returns:
        True when the dataset was committed to ``final_dir``; False when ``write`` declined.

    Raises:
        OSError: If the commit rename fails.
    """
    staging = staging_dir_for(final_dir)
    _remove_if_exists(staging)
    _recover_aside(final_dir)
    staging.mkdir(parents=True)
    if not write(staging):
        shutil.rmtree(staging)
        return False
    _rename_staged(staging, final_dir)
    return True


def _recover_aside(final_dir: Path) -> None:
    """Resolve a leftover ``.old`` directory from an earlier commit before a new one starts.

    With the final name missing, ``.old`` is the only copy of the previous dataset (a crash
    between the move-aside and the staged rename), so it is restored to the final name. With
    the final name present, ``.old`` is a leftover from a finished commit and is deleted.
    """
    aside = aside_dir_for(final_dir)
    if not aside.exists():
        return
    if final_dir.exists():
        shutil.rmtree(aside)
    else:
        _move(aside, final_dir)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _move(source: Path, destination: Path) -> None:
    """Rename ``source`` to ``destination`` with os.replace, retrying transient locks.

    Windows can hold a freshly written file for a moment (antivirus, indexing), so a rename
    may raise PermissionError while nothing is wrong. Only PermissionError is retried; any
    other OSError is reported immediately. The final failure is reported through
    ``_raise_storage_error``.
    """
    for attempt in range(1, RENAME_ATTEMPTS + 1):
        try:
            os.replace(source, destination)
        except PermissionError as exc:
            if attempt == RENAME_ATTEMPTS:
                _raise_storage_error("Committing dataset", destination, exc)
            time.sleep(RENAME_RETRY_DELAY_SECONDS)
        except OSError as exc:
            _raise_storage_error("Committing dataset", destination, exc)
        else:
            return


def _rename_staged(staging: Path, final_dir: Path) -> None:
    """Rename the staging directory to its final name, replacing any existing dataset.

    An existing final directory is first moved aside. If the staged rename then fails, the
    aside copy is moved back so the previous dataset is intact before the error propagates.
    The aside copy is deleted only after the staged directory is in place.
    """
    if not final_dir.exists():
        _move(staging, final_dir)
        return
    aside = aside_dir_for(final_dir)
    _move(final_dir, aside)
    try:
        _move(staging, final_dir)
    except OSError as commit_error:
        try:
            _move(aside, final_dir)
        except OSError as restore_error:
            raise restore_error from commit_error
        raise
    shutil.rmtree(aside)
