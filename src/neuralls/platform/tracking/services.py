"""MLflow-backed tracking services: the one place clients, stores and leases are built."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.artifact_access import MlflowArtifactLeaseManager
from neuralls.platform.tracking.mlflow_client import MlflowClientFactory
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore


@dataclass(frozen=True, slots=True)
class MlflowTrackingServices:
    """Builds the MLflow collaborators composition needs from a tracking URI.

    Attributes:
        client_factory (MlflowClientFactory): Builds every client, including the
            ones inside identity stores and lease managers; tests substitute a
            fake here so no collaborator reaches a real store.
    """

    client_factory: MlflowClientFactory = MlflowClient

    def client(self, tracking_uri: str) -> MlflowClient:
        """Return a client bound to ``tracking_uri``."""
        return self.client_factory(tracking_uri=tracking_uri)

    def identity_store(
        self, *, tracking_uri: str, experiment: str, require_checkpoint: bool = False
    ) -> MlflowIdentityStore:
        """Return the reuse lookup over one experiment of ``tracking_uri``."""
        return MlflowIdentityStore(
            tracking_uri=tracking_uri,
            experiment=experiment,
            require_checkpoint=require_checkpoint,
            client_factory=self.client_factory,
        )

    def lease_manager(self, tracking_uri: str) -> MlflowArtifactLeaseManager:
        """Return a lease manager that resolves artifacts of ``tracking_uri``."""
        return MlflowArtifactLeaseManager(client=self.client(tracking_uri))
