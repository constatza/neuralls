"""Zero-masks over vectors: choose a support, then zero everything outside it.

Choosing the support and applying it are separate functions so another mask
variant (e.g. a contiguous area) only needs a new support function; `apply_mask`
is shared. Nothing here knows where the vector comes from.
"""

from __future__ import annotations

import numpy as np


def random_support_mask(dim: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask keeping ``max(1, round(density * dim))`` entries, drawn without replacement.

    At least one entry is always kept, because a fully zero vector would make
    every downstream system trivial.
    """
    support_size = max(1, round(density * dim))
    mask = np.zeros(dim, dtype=np.bool_)
    mask[rng.choice(dim, size=support_size, replace=False)] = True
    return mask


def apply_mask(vector: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of ``vector`` with every entry outside ``mask`` set to zero."""
    return np.where(mask, vector, 0.0).astype(vector.dtype, copy=False)
