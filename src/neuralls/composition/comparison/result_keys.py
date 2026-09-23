"""Stable result-key invariants for preconditioner comparisons."""

from __future__ import annotations

from collections.abc import Sequence

from neuralls.platform.config.models.preconditioner import PreconditionerConfig


def validate_unique_preconditioner_keys(
    preconditioner_configs: Sequence[PreconditionerConfig],
) -> tuple[str, ...]:
    """Return config keys after proving every requested solve has one unique key."""
    keys = tuple(cfg.name for cfg in preconditioner_configs)
    seen: set[str] = set()
    duplicates: list[str] = []
    for key in keys:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        joined = ", ".join(repr(key) for key in duplicates)
        raise ValueError(
            "Preconditioner names are result identities and must be unique; "
            f"duplicate keys: {joined}."
        )
    return keys
