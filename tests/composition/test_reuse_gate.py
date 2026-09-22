"""`gate_reuse`: force handling and the reuse decision live in exactly one place."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from neuralls.composition.identity.gate import gate_reuse
from neuralls.domain.identity import Reused, StageIdentity
from neuralls.shared.digest import canonical_digest


@dataclass
class FakeStore:
    """In-memory `IdentityStore`: returns what was registered, records lookups."""

    runs: dict[str, str] = field(default_factory=dict)
    lookups: list[StageIdentity] = field(default_factory=list)

    def find(self, identity: StageIdentity) -> Reused | None:
        self.lookups.append(identity)
        run_id = self.runs.get(identity.key)
        return None if run_id is None else Reused(run_id=run_id)


@pytest.fixture
def identity() -> StageIdentity:
    """Identity under test."""
    return StageIdentity.build("training", {"dataset": canonical_digest("d")})


@pytest.fixture
def store(identity: StageIdentity) -> FakeStore:
    """Store holding one run for ``identity``."""
    return FakeStore(runs={identity.key: "run-1"})


def test_hit_returns_the_reused_artifact(store: FakeStore, identity: StageIdentity) -> None:
    assert gate_reuse(store, identity, force=False, label="a") == Reused(run_id="run-1")


def test_miss_returns_none(identity: StageIdentity) -> None:
    assert gate_reuse(FakeStore(), identity, force=False, label="a") is None


def test_force_skips_the_lookup_entirely(store: FakeStore, identity: StageIdentity) -> None:
    """force always executes and never even asks the store."""
    assert gate_reuse(store, identity, force=True, label="a") is None
    assert store.lookups == []


def test_a_different_identity_never_matches(store: FakeStore) -> None:
    other = StageIdentity.build("training", {"dataset": canonical_digest("changed")})
    assert gate_reuse(store, other, force=False, label="a") is None


def test_decisions_are_logged_with_the_reason(
    store: FakeStore, identity: StageIdentity, capfd: pytest.CaptureFixture[str]
) -> None:
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        gate_reuse(store, identity, force=False, label="hit")
        gate_reuse(FakeStore(), identity, force=False, label="miss")
        gate_reuse(store, identity, force=True, label="forced")
    finally:
        logger.remove(sink)
    text = "\n".join(messages)
    assert "reusing run run-1" in text
    assert "no prior run" in text and "dataset" in text
    assert "forced" in text
