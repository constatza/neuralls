"""`run_wall_seconds` reads a run's duration from the store and never invents one."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.mlflow_client import FakeMlflowClient

from neuralls.domain.solver.models.cost import Known, Unavailable
from neuralls.platform.tracking.run_duration import run_wall_seconds
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.types import CostProvenance

START_MS = 1_000
END_MS = 91_500


@pytest.fixture
def fake_client(tmp_path: Path) -> FakeMlflowClient:
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


def test_a_finished_run_costs_its_start_to_end_span(fake_client: FakeMlflowClient) -> None:
    run_id = fake_client.add_run("trained", start_time_ms=START_MS, end_time_ms=END_MS)
    client = MlflowTrackingServices(client_factory=fake_client).client(fake_client.tracking_uri)

    assert run_wall_seconds(client, run_id) == Known(90.5, None, CostProvenance.HISTORICAL)


def test_a_run_that_has_not_ended_is_unavailable(fake_client: FakeMlflowClient) -> None:
    run_id = fake_client.add_run("running", start_time_ms=START_MS)
    client = MlflowTrackingServices(client_factory=fake_client).client(fake_client.tracking_uri)

    assert run_wall_seconds(client, run_id) == Unavailable(f"run {run_id} has no end time")


def test_a_run_the_store_does_not_hold_is_unavailable(fake_client: FakeMlflowClient) -> None:
    client = MlflowTrackingServices(client_factory=fake_client).client(fake_client.tracking_uri)

    assert run_wall_seconds(client, "missing") == Unavailable(
        "run missing cannot be read: Run 'missing' not found"
    )
