"""Tests for strict model_ref resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments.model_resolution import (
    ModelResolution,
    PreconditionerResolutionResult,
    resolve_model_ref,
    resolve_preconditioner_models,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.platform.config.models.checkpoint_refs import (
    ByAlias,
    ByRunId,
    LoggedModelRefConfig,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    NoopArtifactLeaseManager,
)
from neuralls.platform.tracking.model_registry import (
    CHECKPOINT_ARTIFACT_PATH_TAG,
)
from neuralls.platform.tracking.services import MlflowTrackingServices
from tests.fakes.checkpoint_lease_manager import CheckpointLeaseManager

# Never opened: every test patches the MLflow-facing collaborators.
TRACKING_URI = "sqlite:///workspace/mlflow.db"
WORKSPACE = Path("workspace")


@pytest.fixture
def noop_artifact_leases() -> ArtifactLeaseManager:
    """Lease manager for tests that must not resolve MLflow artifacts."""
    return NoopArtifactLeaseManager()


def test_resolve_preconditioner_models_keeps_non_neural_specs(
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Non-neural preconditioners pass through unchanged."""
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    resolved = resolve_preconditioner_models(
        specs=[jacobi],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
    )
    assert resolved == [jacobi]


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_sets_resolved_checkpoint(
    mock_resolve_model_ref,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs receive resolved_checkpoint_path from resolver."""
    checkpoint = WORKSPACE / "resolved.ckpt"
    mock_resolve_model_ref.return_value = ModelResolution(
        model_uri="runs:/run-1/model",
        run_id="run-1",
        checkpoint_path=checkpoint,
    )

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="run-1")),
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
    )
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.type == PreconditionerType.NEURAL
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_keeps_explicit_checkpoint(
    mock_resolve_model_ref,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs with explicit checkpoint_path bypass model_ref resolution."""
    checkpoint = WORKSPACE / "explicit.ckpt"
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint,
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
    )

    mock_resolve_model_ref.assert_not_called()
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


def test_resolve_preconditioner_models_requires_model_source(
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs without checkpoint_path/model_ref are invalid for resolution."""
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
    )

    with pytest.raises(ValueError):
        resolve_preconditioner_models(
            specs=[neural],
            tracking_uri=TRACKING_URI,
            artifact_leases=noop_artifact_leases,
        )


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_with_warnings_skips_unresolved_neural(
    mock_resolve_model_ref,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Comparison-specific resolution can skip unresolved neural specs."""
    mock_resolve_model_ref.side_effect = ValueError("Registered model 'MissingFFNN' not found")
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="MissingFFNN", selector=ByAlias(alias="solutions")),
    )

    resolved = resolve_preconditioner_models_with_warnings(
        specs=[jacobi, neural],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
        skip_unresolved=True,
    )

    assert isinstance(resolved, PreconditionerResolutionResult)
    assert resolved.specs == [jacobi]
    assert len(resolved.warnings) == 1


def test_resolve_model_ref_dataset_placeholder_requires_dataset_alias(
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Using alias=@dataset requires a provided dataset alias context."""
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            selector=ByAlias(alias="@dataset"),
        ),
    )
    with pytest.raises(ValueError):
        resolve_model_ref(
            spec=neural,
            tracking_uri=TRACKING_URI,
            artifact_leases=noop_artifact_leases,
            tracking=MlflowTrackingServices(client_factory=MagicMock()),
        )


def test_resolve_model_ref_rejects_mismatched_artifact_lease_store() -> None:
    """Model metadata and artifact leases must come from the same tracking store."""

    class MismatchedLeaseManager(NoopArtifactLeaseManager):
        @property
        def tracking_uri(self) -> str | None:
            return "sqlite:////tmp/other-mlflow.db"

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="run-1")),
    )

    with pytest.raises(ValueError):
        resolve_model_ref(
            spec=neural,
            tracking_uri=TRACKING_URI,
            artifact_leases=MismatchedLeaseManager(),
        )


def test_resolve_model_ref_normalizes_explicit_at_alias() -> None:
    """Registered alias values strip @ prefix before MLflow lookup."""
    client = MagicMock()
    version = client.get_model_version_by_alias.return_value
    version.run_id = "run-1"
    version.version = "1"
    version.tags = {CHECKPOINT_ARTIFACT_PATH_TAG: "checkpoints/best.ckpt"}
    artifact_leases = CheckpointLeaseManager(checkpoint_root=WORKSPACE)

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            selector=ByAlias(alias="@solutions"),
        ),
    )
    resolve_model_ref(
        spec=neural,
        tracking_uri=TRACKING_URI,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=MagicMock(return_value=client)),
    )

    client.get_model_version_by_alias.assert_called_once_with(
        "NormScaledLinearFFNN",
        "solutions",
    )
    assert artifact_leases.file_calls == [("run-1", "checkpoints/best.ckpt")]
