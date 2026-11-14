"""Drive ``run_assignment`` with every input but the reuse decision patched.

Shared by the in-memory reuse tests and the real-store reuse tests, which differ only in the
tracking services they pass.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuralls.application.models import AssignmentResult
from neuralls.composition.assignments import training_batch
from neuralls.composition.assignments.training import PreparedTraining
from neuralls.composition.tracking.services import TrackingServices
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.config.models.workspace import AssignmentSpec
from neuralls.shared.digest import canonical_digest

OUTPUT_ROOT = Path("workspace/output")
"""Never read: every collaborator that would touch it is patched."""

EXPERIMENT = "Train"

TRACKING_URI = "sqlite:///workspace/tracking.db"
"""Never opened by the in-memory tests: their tracking services are mocks."""

IDENTITY = StageIdentity.build("training", {"dataset": canonical_digest("d")})
"""The training identity the patched derivation returns."""

OTHER_IDENTITY = StageIdentity.build("training", {"dataset": canonical_digest("other")})
"""The identity of an assignment trained on different inputs."""


@dataclass(frozen=True, slots=True)
class PatchedInputs:
    """The patched identity derivation and training preparation a test steers and inspects."""

    derive: MagicMock
    prepare_training_settings: MagicMock


@contextmanager
def patched_assignment_inputs(identity: StageIdentity) -> Generator[PatchedInputs]:
    """Patch dataset loading, identity derivation and preparation; dataset files all exist."""
    present = MagicMock(spec=Path)
    present.exists.return_value = True
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "resolve_dataset_artifacts") as resolve_dataset_artifacts,
        patch.object(training_batch, "training_identity", return_value=identity) as derive,
        patch.object(training_batch, "prepare_training_settings") as prepare_training_settings,
    ):
        data_cfg = MagicMock()
        data_cfg.id = "assignment-dataset"
        data_cfg.output.data_dir = OUTPUT_ROOT
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"
        resolve_dataset_artifacts.return_value = MagicMock(
            rhs=MagicMock(path=present),
            solutions=MagicMock(path=present),
            matrix=MagicMock(path=present),
        )
        prepare_training_settings.return_value = MagicMock(name="prepared-training")
        yield PatchedInputs(derive, prepare_training_settings)


def assignment_spec(*, assignment_id: str = "exp-1") -> AssignmentSpec:
    """Minimal assignment spec naming the job/data configs run_assignment reads."""
    return AssignmentSpec(
        assignment_id=assignment_id,
        job_config_path=Path("workspace/job.toml"),
        data_config_path=Path("workspace/data.toml"),
    )


def run_once(
    presentation: TrainingPresentation,
    *,
    assignment_id: str = "exp-1",
    tracking: TrackingServices,
    tracking_uri: str,
    force: bool = False,
) -> AssignmentResult | PreparedTraining:
    return training_batch.run_assignment(
        settings=MagicMock(),
        spec=assignment_spec(assignment_id=assignment_id),
        output_root=OUTPUT_ROOT,
        force=force,
        presentation=presentation,
        mlflow_experiment_name=EXPERIMENT,
        tracking_uri=tracking_uri,
        tracking=tracking,
    )
