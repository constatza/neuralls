"""`MlflowIdentityStore`: exact-key, experiment-scoped, deterministic run lookup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from mlflow.tracking import MlflowClient

from neuralls.domain.identity import IdentityTag, Reused, StageIdentity
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore
from neuralls.shared.digest import canonical_digest

EXPERIMENT = "Train-Case"

type RunFactory = Callable[..., str]


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Isolated SQLite MLflow store."""
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture
def client(tracking_uri: str) -> MlflowClient:
    """Client bound to the isolated store."""
    return MlflowClient(tracking_uri=tracking_uri)


@pytest.fixture
def identity() -> StageIdentity:
    """Identity under test."""
    return StageIdentity.build(
        "training", {"dataset": canonical_digest("d"), "job": canonical_digest("j")}
    )


@pytest.fixture
def other_identity() -> StageIdentity:
    """Identity differing in one component."""
    return StageIdentity.build(
        "training", {"dataset": canonical_digest("d"), "job": canonical_digest("J2")}
    )


@pytest.fixture
def make_run(client: MlflowClient) -> RunFactory:
    """Create a run with explicit start time, status and tags in a named experiment."""

    def _make(
        identity: StageIdentity,
        *,
        start: int,
        status: str = "FINISHED",
        experiment: str = EXPERIMENT,
    ) -> str:
        found = client.get_experiment_by_name(experiment)
        experiment_id = found.experiment_id if found else client.create_experiment(experiment)
        run = client.create_run(experiment_id, start_time=start, tags=identity.tags())
        client.set_terminated(run.info.run_id, status=status, end_time=start + 1)
        return run.info.run_id

    return _make


@pytest.fixture
def store(tracking_uri: str) -> MlflowIdentityStore:
    """Store for the training experiment, no checkpoint requirement."""
    return MlflowIdentityStore(tracking_uri=tracking_uri, experiment=EXPERIMENT)


def test_missing_experiment_is_a_miss(store: MlflowIdentityStore, identity: StageIdentity) -> None:
    """No experiment means nothing to reuse."""
    assert store.find(identity) is None


def test_newest_finished_run_with_the_exact_key_wins(
    store: MlflowIdentityStore, identity: StageIdentity, make_run: RunFactory
) -> None:
    """Among equal-key runs the newest wins."""
    make_run(identity, start=1000)
    newest = make_run(identity, start=2000)
    assert store.find(identity) == Reused(run_id=newest)


def test_other_key_failed_and_other_experiment_runs_never_match(
    store: MlflowIdentityStore,
    identity: StageIdentity,
    other_identity: StageIdentity,
    make_run: RunFactory,
) -> None:
    """Different key, non-FINISHED status and other experiments are all ignored."""
    good = make_run(identity, start=1000)
    make_run(other_identity, start=5000)
    make_run(identity, start=6000, status="FAILED")
    make_run(identity, start=7000, experiment="Other")
    assert store.find(identity) == Reused(run_id=good)


def test_run_without_checkpoint_falls_back_to_older_run(
    tracking_uri: str, identity: StageIdentity, make_run: RunFactory
) -> None:
    """A newer FINISHED run lacking a checkpoint does not shadow an older complete one."""
    older = make_run(identity, start=1000)
    make_run(identity, start=2000)
    store = MlflowIdentityStore(
        tracking_uri=tracking_uri, experiment=EXPERIMENT, require_checkpoint=True
    )
    with patch(
        "neuralls.platform.tracking.mlflow_store.has_checkpoint_artifact",
        side_effect=lambda run_id, **_: run_id == older,
    ):
        assert store.find(identity) == Reused(run_id=older)


def test_walk_is_not_capped_by_page_size(
    store: MlflowIdentityStore, tracking_uri: str, identity: StageIdentity, make_run: RunFactory
) -> None:
    """More broken newer runs than one page cannot hide an older good run."""
    good = make_run(identity, start=1)
    for offset in range(3):
        make_run(identity, start=100 + offset)
    strict = MlflowIdentityStore(
        tracking_uri=tracking_uri, experiment=EXPERIMENT, require_checkpoint=True
    )
    with (
        patch("neuralls.platform.tracking.mlflow_store._PAGE_SIZE", 2),
        patch(
            "neuralls.platform.tracking.mlflow_store.has_checkpoint_artifact",
            side_effect=lambda run_id, **_: run_id == good,
        ),
    ):
        assert strict.find(identity) == Reused(run_id=good)


def test_identity_tags_use_the_shared_enum(identity: StageIdentity) -> None:
    """Written tag names are exactly the enum members the store filters on."""
    assert {IdentityTag.STAGE, IdentityTag.KEY, IdentityTag.COMPONENTS} == set(identity.tags())
