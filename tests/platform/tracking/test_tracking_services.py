"""MlflowTrackingServices builds every collaborator from its one client factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.mlflow_client import FakeMlflowClient

from neuralls.platform.tracking.services import MlflowTrackingServices


@pytest.fixture
def fake_client(tmp_path: Path) -> FakeMlflowClient:
    """In-memory MLflow store whose run artifacts live under ``tmp_path``."""
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


@pytest.fixture
def services(fake_client: FakeMlflowClient) -> MlflowTrackingServices:
    """Services whose every client is the fake store."""
    return MlflowTrackingServices(client_factory=lambda **_: fake_client)


def test_client_comes_from_the_factory(
    services: MlflowTrackingServices, fake_client: FakeMlflowClient
) -> None:
    assert services.client("sqlite:///unused") is fake_client


def test_identity_store_is_scoped_to_the_requested_experiment(
    services: MlflowTrackingServices,
) -> None:
    store = services.identity_store(
        tracking_uri="sqlite:///unused", experiment="Train", require_checkpoint=True
    )

    assert (store.tracking_uri, store.experiment, store.require_checkpoint) == (
        "sqlite:///unused",
        "Train",
        True,
    )
    assert store.client_factory is services.client_factory


def test_lease_manager_is_bound_to_the_factory_client(
    services: MlflowTrackingServices, fake_client: FakeMlflowClient
) -> None:
    with services.lease_manager("sqlite:///unused") as leases:
        assert leases.tracking_uri == str(fake_client.tracking_uri)
