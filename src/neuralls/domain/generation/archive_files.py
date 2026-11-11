"""Archive file selection and cheap row counting, for file-backed generation strategies."""

from __future__ import annotations

from pathlib import Path

from .ports import ArchiveSource
from .seeds import rng_from_seed


def select_archive_files(
    source: ArchiveSource,
    glob_pattern: str,
    count: int,
    shuffle: bool,
    seed: int | None,
    skip: int = 0,
    file_indices: tuple[int, ...] | None = None,
) -> list:
    """Select N files from glob pattern with optional shuffling.

    Pure function for deterministic file selection from archives.

    Args:
        source: Lists the archive's files.
        glob_pattern: Path pattern like "/data/*.txt" or "/data/rhs_*.npy"
        count: Number of files to select (-1 means all available files)
        shuffle: Whether to shuffle before selection
        seed: Random seed for shuffling (required if shuffle=True)
        skip: Number of files to skip after deterministic ordering/shuffling
        file_indices: Explicit positions in the pool that remains after ``skip``. When
            given, ``count`` must equal their number and exactly these files are returned,
            in this order. Lets the caller give each binding a disjoint slice of one pool.

    Returns:
        List of selected file paths (pathlib.Path objects)

    Raises:
        FileNotFoundError: If no files match pattern or directory doesn't exist
        ValueError: If count and skip exceed available files

    Examples:
        >>> # Select first 10 files
        >>> files = select_archive_files(source, "/data/rhs_*.txt", 10, False, None)

        >>> # Select all files with shuffling
        >>> files = select_archive_files(source, "/data/sol_*.npy", -1, True, 42)

        >>> # Select 50 shuffled files
        >>> files = select_archive_files(source, "/data/vec_*.txt", 50, True, 42)
    """
    pattern_path = Path(glob_pattern)
    directory = pattern_path.parent
    pattern = pattern_path.name

    # Validate directory exists
    if not source.exists(directory):
        raise FileNotFoundError(f"Archive directory not found: {directory}")

    # Scan for matching files
    candidates = source.glob(directory, pattern)
    if not candidates:
        raise FileNotFoundError(f"No files found matching pattern: {directory / pattern}")

    if skip < 0:
        raise ValueError(f"Archive skip must be non-negative, got {skip}")
    if skip > len(candidates):
        raise ValueError(
            f"Requested skip={skip} but only {len(candidates)} files are available "
            f"matching pattern: {glob_pattern}"
        )

    ordered = _ordered_candidates(candidates, shuffle, seed)
    if file_indices is not None:
        return _pick_explicit_files(ordered[skip:], count, file_indices, glob_pattern)

    # Handle "all files" case
    if count == -1:
        count = len(candidates) - skip

    # Validate sufficient files available
    if skip + count > len(candidates):
        if skip == 0:
            raise ValueError(
                f"Requested {count} files but only {len(candidates)} available "
                f"matching pattern: {glob_pattern}"
            )
        raise ValueError(
            f"Requested {count} files with skip={skip} but only {len(candidates)} available "
            f"matching pattern: {glob_pattern}"
        )

    return ordered[skip : skip + count]


def _ordered_candidates(candidates: list, shuffle: bool, seed: int | None) -> list:
    """Return the sorted candidates, permuted by ``seed`` when ``shuffle`` is set."""
    if not shuffle:
        return candidates
    rng = rng_from_seed(seed)
    return [candidates[idx] for idx in rng.permutation(len(candidates))]


def _pick_explicit_files(
    pool: list,
    count: int,
    file_indices: tuple[int, ...],
    glob_pattern: str,
) -> list:
    """Return the pool entries at ``file_indices`` after checking they form a valid selection."""
    if count != len(file_indices):
        raise ValueError(
            f"count={count} must equal the {len(file_indices)} explicit file indices "
            f"matching pattern: {glob_pattern}"
        )
    out_of_range = [idx for idx in file_indices if not 0 <= idx < len(pool)]
    if out_of_range:
        raise ValueError(
            f"Explicit file indices {out_of_range} are outside the {len(pool)} files "
            f"available matching pattern: {glob_pattern}"
        )
    return [pool[idx] for idx in file_indices]
