"""Port for the tracking collaborators composition builds from a tracking URI."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from mlflow.tracking import MlflowClient

from neuralls.domain.identity_ports import IdentityStore
from neuralls.platform.tracking.artifact_access import ArtifactLeaseManager
from neuralls.platform.tracking.services import MlflowTrackingServices


class TrackingServices(Protocol):
    """Factory for the clients, reuse lookups and artifact leases of one tracking store.

    Composition asks this port instead of constructing MLflow objects, so the
    entry point picks the backend once and tests substitute fakes.
    """

    def client(self, tracking_uri: str) -> MlflowClient:
        """Return a client bound to ``tracking_uri``."""
        ...

    def identity_store(
        self, *, tracking_uri: str, experiment: str, require_checkpoint: bool = False
    ) -> IdentityStore:
        """Return the reuse lookup over one experiment (``require_checkpoint`` for training runs)."""
        ...

    def lease_manager(self, tracking_uri: str) -> AbstractContextManager[ArtifactLeaseManager]:
        """Return a lease manager that resolves artifacts of ``tracking_uri``."""
        ...


# Stateless and immutable: entry points default to it so only the CLI and tests choose otherwise.
MLFLOW_TRACKING: TrackingServices = MlflowTrackingServices()
