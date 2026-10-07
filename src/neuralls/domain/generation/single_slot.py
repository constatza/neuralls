"""An explicit one-entry cache: keeps only the most recently requested key's value.

Both the matrix loader (``matrix_cache.py``) and the per-binding input loader
(``orchestration.py``) visit keys in order and only need the current one kept alive, so a
plain ``lru_cache(maxsize=1)`` closure would do the same job — but it hides the "one slot,
most recent wins" behavior behind a decorator and depends on call order silently. This makes
that behavior an explicit, readable object instead.
"""

from __future__ import annotations

from collections.abc import Callable


class SingleSlot[K, V]:
    """Caches the most recent ``(key, value)`` pair; any other key replaces it."""

    def __init__(self) -> None:
        self._entry: tuple[K, V] | None = None

    def get(self, key: K, loader: Callable[[K], V]) -> V:
        """Return the cached value for ``key``, loading and replacing the slot otherwise."""
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]
        value = loader(key)
        self._entry = (key, value)
        return value
