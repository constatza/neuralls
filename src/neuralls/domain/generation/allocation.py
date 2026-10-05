"""Pure sample-budget allocation across matrices (no I/O).

A budget of ``total`` units over ``num_matrices`` matrices is split so that every
matrix gets the same base share and the leftover units go to distinct matrices.
Nothing is dropped: the per-matrix counts always sum to ``total``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def split_remainder(total: int, num_matrices: int, *, seed: int) -> NDArray[np.int64]:
    """Split ``total`` units over ``num_matrices`` matrices with one remainder unit each.

    Each matrix gets ``total // num_matrices`` units. The ``total % num_matrices``
    leftover units go to distinct matrices chosen uniformly at random from a
    generator seeded with ``seed``, so no matrix receives more than one extra unit
    and the choice is reproducible for a given seed.

    Args:
        total: Non-negative budget to distribute.
        num_matrices: Number of matrices, at least 1.
        seed: Seed for the generator that picks the matrices receiving a leftover unit.

    Returns:
        Integer array of length ``num_matrices`` whose entries sum to ``total``.

    Raises:
        ValueError: If ``total`` is negative or ``num_matrices`` is below 1.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if num_matrices < 1:
        raise ValueError(f"num_matrices must be at least 1, got {num_matrices}")

    base, remainder = divmod(total, num_matrices)
    counts = np.full(num_matrices, base, dtype=np.int64)
    rng = np.random.default_rng(seed)
    extra = rng.choice(num_matrices, size=remainder, replace=False)
    counts[extra] += 1
    return counts


def archive_units(
    num_matrices: int,
    num_files: int,
    count: int,
) -> tuple[tuple[int, int], ...]:
    """Map unit positions ``t`` onto (matrix, file) pairs with a cyclic rule.

    For ``t`` in ``range(min(count, num_matrices * num_files))``: the matrix is
    ``t % num_matrices`` and the file is ``(matrix + t // num_matrices) % num_files``.
    Consecutive units rotate through matrices first, so each matrix's files advance
    by one per pass over the matrices. Until the grid is exhausted, no (matrix, file)
    pair repeats, and matrices receive units as evenly as the count allows.

    The cap at ``num_matrices * num_files`` is part of the map itself: the function
    emits exactly that many pairs for larger requests. Warning about the cap is the
    caller's job.

    Args:
        num_matrices: Number of matrices, at least 1.
        num_files: Number of archive files per matrix, at least 1.
        count: Requested number of units, non-negative.

    Returns:
        Pairs ``(matrix_index, file_index)`` in unit order.

    Raises:
        ValueError: If ``num_matrices`` or ``num_files`` is below 1, or ``count`` is negative.
    """
    if num_matrices < 1:
        raise ValueError(f"num_matrices must be at least 1, got {num_matrices}")
    if num_files < 1:
        raise ValueError(f"num_files must be at least 1, got {num_files}")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    emitted = min(count, num_matrices * num_files)
    pairs: list[tuple[int, int]] = []
    for position in range(emitted):
        matrix, pass_index = position % num_matrices, position // num_matrices
        pairs.append((matrix, (matrix + pass_index) % num_files))
    return tuple(pairs)
