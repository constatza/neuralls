"""Ports for identity-based reuse (implementations live in ``platform``)."""

from __future__ import annotations

from typing import Protocol

from neuralls.domain.identity import Reused, StageIdentity


class IdentityStore(Protocol):
    """Finds an artifact previously produced under an identical identity."""

    def find(self, identity: StageIdentity) -> Reused | None:
        """Return the reusable artifact for ``identity``, or ``None`` on a miss."""
        ...
