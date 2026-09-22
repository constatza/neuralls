"""Strict, unambiguous resolution of an assignment's training run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from mlflow.exceptions import MlflowException

from neuralls.composition.assignments import model_resolution
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    TrainingLookup,
    resolve_model_ref,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.domain.identity import Reused, StageIdentity
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.tracking.artifact_access import NoopArtifactLeaseManager
from neuralls.shared.digest import canonical_digest

ASSIGNMENT_ID = "pod-2g_rank10"


@pytest.fixture
def ref() -> TrainedAssignmentRefConfig:
    """Strict ref to the assignment under test."""
    return TrainedAssignmentRefConfig(assignment_id=ASSIGNMENT_ID)


@pytest.fixture
def identity() -> StageIdentity:
    """The assignment's current derived training identity."""
    return StageIdentity.build("training", {"dataset": canonical_digest("d")})


@pytest.fixture
def lookup(identity: StageIdentity) -> TrainingLookup:
    """Training experiment plus the current training identity."""
    return TrainingLookup(experiment="Train-Case", identity=identity)


@pytest.fixture
def store_cls():
    """Patched identity store class used by the strict resolver."""
    with patch.object(model_resolution, "MlflowIdentityStore") as cls:
        yield cls


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A checkpoint path standing in for the leased artifact."""
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"")
    return path


@pytest.fixture
def neural_spec(ref: TrainedAssignmentRefConfig) -> NeuralPreconditionerConfig:
    """Neural preconditioner stub referencing the assignment's training run."""
    return NeuralPreconditionerConfig(
        name="stub", type=PreconditionerType.NEURAL, assignment=ASSIGNMENT_ID, model_ref=ref
    )


def _resolve(ref: TrainedAssignmentRefConfig, lookup: TrainingLookup | None):
    return resolve_model_ref(
        spec=NeuralPreconditionerConfig(
            name="stub", type=PreconditionerType.NEURAL, assignment=ASSIGNMENT_ID, model_ref=ref
        ),
        tracking_uri="sqlite:///unused",
        artifact_leases=NoopArtifactLeaseManager(),
        training=lookup,
    )


def test_lookup_is_scoped_to_experiment_and_exact_identity(
    ref: TrainedAssignmentRefConfig,
    lookup: TrainingLookup,
    identity: StageIdentity,
    checkpoint: Path,
    store_cls,
) -> None:
    """The store is built for the training experiment, requires a checkpoint, and is
    asked for exactly the assignment's identity."""
    store_cls.return_value.find.return_value = Reused(run_id="run-1")
    with patch.object(model_resolution, "_resolve_checkpoint_for_run", return_value=checkpoint):
        resolution = _resolve(ref, lookup)

    store_cls.assert_called_once_with(
        tracking_uri="sqlite:///unused", experiment="Train-Case", require_checkpoint=True
    )
    store_cls.return_value.find.assert_called_once_with(identity)
    assert (resolution.run_id, resolution.checkpoint_path) == ("run-1", checkpoint)


def test_no_matching_run_is_an_error_not_a_fallback(
    ref: TrainedAssignmentRefConfig, lookup: TrainingLookup, store_cls
) -> None:
    """When nothing matches the current inputs, resolution raises with guidance."""
    store_cls.return_value.find.return_value = None
    with pytest.raises(ValueError, match="train this assignment first"):
        _resolve(ref, lookup)


def test_missing_identity_fails_without_searching(
    ref: TrainedAssignmentRefConfig, store_cls
) -> None:
    """An ungenerated dataset cannot be verified, so no search is attempted."""
    with pytest.raises(ValueError, match="not generated"):
        _resolve(ref, TrainingLookup(experiment="Train-Case", identity=None))
    store_cls.assert_not_called()


def test_missing_training_context_is_an_error(ref: TrainedAssignmentRefConfig) -> None:
    """A ref resolved without any assignment context raises instead of guessing."""
    with pytest.raises(ValueError, match="no training lookup"):
        _resolve(ref, None)


def test_tracking_errors_skip_one_model_instead_of_aborting(
    neural_spec: NeuralPreconditionerConfig, lookup: TrainingLookup, store_cls
) -> None:
    """An MLflow failure for one model is reported as a warning when skipping is allowed."""
    store_cls.return_value.find.side_effect = MlflowException("server down")
    if True:
        result = resolve_preconditioner_models_with_warnings(
            specs=[neural_spec],
            tracking_uri="sqlite:///unused",
            artifact_leases=NoopArtifactLeaseManager(),
            assignment_contexts={ASSIGNMENT_ID: AssignmentModelContext(training=lookup)},
            skip_unresolved=True,
        )

    assert result.specs == []
    assert len(result.warnings) == 1 and "server down" in result.warnings[0]
