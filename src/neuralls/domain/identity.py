"""Stage identity value objects: what makes a generated/trained/compared artifact reusable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from neuralls.shared.digest import Digest, canonical_digest

_SHORT_DIGEST_LENGTH = len("sha256:") + 16


class IdentityTag(StrEnum):
    """Every persisted identity field name, for both writing and filtering.

    One enum is the only place these strings exist, so a tag can never be
    written without a lookup that enforces it (or vice versa).
    """

    STAGE = "neuralls.identity.stage"
    KEY = "neuralls.identity.key"
    COMPONENTS = "neuralls.identity.components"
    CHECKPOINT_DIGEST = "neuralls.identity.checkpoint_digest"
    ENV = "neuralls.identity.env"


@dataclass(frozen=True)
class StageIdentity:
    """Derived identity of one stage's output.

    Attributes:
        stage (str): Stage name (e.g. ``"training"``); part of the key.
        key (Digest): Digest over the stage name and every component digest.
        components (Mapping[str, str]): Short per-input digests, kept only to
            explain *which* input changed when a lookup misses.
    """

    stage: str
    key: Digest
    components: Mapping[str, str]

    @classmethod
    def build(cls, stage: str, components: Mapping[str, Digest]) -> StageIdentity:
        """Derive an identity from named input digests.

        Args:
            stage (str): Stage name.
            components (Mapping[str, Digest]): Full digest of every input that
                determines the output (config, upstream identities, data, ...).

        Returns:
            StageIdentity: Identity whose key changes iff any component changes.
        """
        key = canonical_digest(stage, dict(components))
        short = {name: digest[:_SHORT_DIGEST_LENGTH] for name, digest in components.items()}
        return cls(stage=stage, key=key, components=short)

    def tags(self) -> dict[str, str]:
        """Return the string tags to write on an MLflow run."""
        return {
            IdentityTag.STAGE: self.stage,
            IdentityTag.KEY: self.key,
            IdentityTag.COMPONENTS: json.dumps(dict(self.components), sort_keys=True),
        }

    def manifest_fields(self) -> dict[str, object]:
        """Return the fields to persist in a dataset manifest."""
        return {"identity_key": self.key, "identity_components": dict(self.components)}


@dataclass(frozen=True)
class Reused:
    """A previously produced artifact whose identity matched.

    Attributes:
        run_id (str | None): Producing MLflow run, or ``None`` for artifacts
            that live outside MLflow (datasets).
    """

    run_id: str | None = None
