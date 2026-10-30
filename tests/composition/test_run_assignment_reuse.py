"""MLflow-backed reuse: an assignment only skips retraining if the identity store
finds a FINISHED run with a checkpoint carrying the assignment's derived training
identity (dataset content, effective job settings, epoch override) — never based on
a local checkpoint, an assignment label, or another assignment's run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlflow.tracking import MlflowClient

from neuralls.application.models import (
    AssignmentFailed,
    AssignmentResult,
    AssignmentSucceeded,
)
from neuralls.composition.assignments import training_batch
from neuralls.composition.assignments.training import PreparedTraining
from neuralls.composition.tracking.services import MLFLOW_TRACKING, TrackingServices
from neuralls.domain.identity import Reused, StageIdentity
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.config.models.workspace import AssignmentSpec
from neuralls.platform.tracking.run_names import render_training_label
from neuralls.shared.digest import canonical_digest

pytestmark = pytest.mark.deferred_io("mlflow-run-tracking")

type CapturedLog = Callable[[str], list[str]]


def _spec(*, assignment_id: str = "exp-1") -> AssignmentSpec:
    """Minimal assignment spec naming the job/data configs run_assignment reads."""
    return AssignmentSpec(
        assignment_id=assignment_id,
        job_config_path=Path("/workspace/job.toml"),
        data_config_path=Path("/workspace/data.toml"),
    )


@pytest.fixture
def identity() -> StageIdentity:
    """Training identity the patched derivation returns."""
    return StageIdentity.build("training", {"dataset": canonical_digest("d")})


@pytest.fixture
def other_identity() -> StageIdentity:
    """Identity of an assignment trained on different inputs."""
    return StageIdentity.build("training", {"dataset": canonical_digest("other")})


@pytest.fixture
def run_assignment_dependencies(
    tmp_path: Path, identity: StageIdentity
) -> Iterator[dict[str, MagicMock]]:
    """Patch every I/O dependency of ``run_assignment`` except the reuse decision."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tracking = MagicMock()
    store_cls = tracking.identity_store
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "resolve_dataset_artifacts") as resolve_dataset_artifacts,
        patch.object(training_batch, "training_identity", return_value=identity) as derive,
        patch.object(training_batch, "prepare_training_settings") as prepare_training_settings,
    ):
        data_cfg = MagicMock()
        data_cfg.id = "assignment-dataset"
        data_cfg.output.data_dir = tmp_path
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"
        resolve_dataset_artifacts.return_value = MagicMock(
            rhs=MagicMock(path=data_dir),
            solutions=MagicMock(path=data_dir),
            matrix=MagicMock(path=data_dir),
        )
        prepare_training_settings.return_value = MagicMock(name="prepared-training")
        store_cls.return_value.find.return_value = None
        yield {
            "find": store_cls.return_value.find,
            "store_cls": store_cls,
            "tracking": tracking,
            "derive": derive,
            "prepare_training_settings": prepare_training_settings,
        }


def _run(
    tmp_path: Path,
    presentation: TrainingPresentation,
    *,
    assignment_id: str = "exp-1",
    force: bool = False,
    tracking_uri: str = "sqlite:///tracking.db",
    tracking: TrackingServices = MLFLOW_TRACKING,
) -> AssignmentResult | PreparedTraining:
    return training_batch.run_assignment(
        settings=MagicMock(),
        spec=_spec(assignment_id=assignment_id),
        output_root=tmp_path,
        force=force,
        presentation=presentation,
        mlflow_experiment_name="Train",
        tracking_uri=tracking_uri,
        tracking=tracking,
    )


def test_train_skips_when_an_identical_run_already_exists(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    identity: StageIdentity,
    training_presentation: TrainingPresentation,
) -> None:
    """A FINISHED run carrying this exact identity short-circuits training."""
    run_assignment_dependencies["find"].return_value = Reused(run_id="existing-run-id")
    result = _run(tmp_path, training_presentation, tracking=run_assignment_dependencies["tracking"])
    run_assignment_dependencies["prepare_training_settings"].assert_not_called()
    run_assignment_dependencies["find"].assert_called_once_with(identity)
    assert isinstance(result, AssignmentSucceeded)
    assert result.status == "Success"
    assert result.mlflow_run_id == "existing-run-id"


def test_lookup_is_scoped_to_the_training_experiment_and_requires_a_checkpoint(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """Reuse only considers the case's training experiment and real checkpoints."""
    _run(tmp_path, training_presentation, tracking=run_assignment_dependencies["tracking"])
    run_assignment_dependencies["store_cls"].assert_called_once_with(
        tracking_uri="sqlite:///tracking.db", experiment="Train", require_checkpoint=True
    )


def test_train_runs_and_tags_the_identity_when_no_matching_run_exists(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    identity: StageIdentity,
    training_presentation: TrainingPresentation,
) -> None:
    """A miss prepares training and stamps the derived identity on the new run."""
    result = _run(tmp_path, training_presentation, tracking=run_assignment_dependencies["tracking"])
    prepare = run_assignment_dependencies["prepare_training_settings"]
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["extra_tags"] == identity.tags()
    assert result is prepare.return_value


def test_force_always_retrains_even_with_a_matching_run(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """force=True bypasses the reuse check entirely."""
    run_assignment_dependencies["find"].return_value = Reused(run_id="existing-run-id")
    result = _run(
        tmp_path,
        training_presentation,
        force=True,
        tracking=run_assignment_dependencies["tracking"],
    )
    run_assignment_dependencies["prepare_training_settings"].assert_called_once()
    run_assignment_dependencies["find"].assert_not_called()
    assert result is run_assignment_dependencies["prepare_training_settings"].return_value


def test_run_assignment_returns_failed_result_on_unexpected_exception(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """An unexpected exception (e.g. dlkit's leaked-MLflow-run error on a search job)
    must be recorded as a Failed result, not raised — run_assignment_sweep's
    "failed assignments don't stop the batch" guarantee depends on this.
    """
    run_assignment_dependencies["prepare_training_settings"].side_effect = Exception(
        "Run with UUID abc123 is already active."
    )
    result = _run(
        tmp_path,
        training_presentation,
        assignment_id="search-job-2",
        tracking=run_assignment_dependencies["tracking"],
    )
    assert isinstance(result, AssignmentFailed)
    assert result.status == "Failed"
    assert result.error is not None


def test_run_assignment_fails_cleanly_when_dataset_was_never_generated(
    tmp_path: Path,
    training_presentation: TrainingPresentation,
) -> None:
    """Training must never auto-generate a missing dataset — that's the generate
    stage's exclusive job. If nobody ran `generate` first, run_assignment must
    fail cleanly (not silently regenerate) with an error pointing at the missing
    dataset, and must never reach training preparation.
    """
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "prepare_training_settings") as prepare_training_settings,
    ):
        data_cfg = MagicMock()
        data_cfg.id = "never-generated"
        data_cfg.output.data_dir = tmp_path / "data"  # never created — dataset never ran
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"

        result = training_batch.run_assignment(
            settings=MagicMock(),
            spec=_spec(assignment_id="exp-1"),
            output_root=tmp_path,
            force=False,
            presentation=training_presentation,
            mlflow_experiment_name="Train",
            tracking_uri="sqlite:///tracking.db",
        )

    assert isinstance(result, AssignmentFailed)
    assert result.status == "Failed"
    assert result.error is not None
    prepare_training_settings.assert_not_called()


def test_a_failed_assignment_is_logged_by_its_training_presentation(
    tmp_path: Path, training_presentation: TrainingPresentation, captured_log: CapturedLog
) -> None:
    """The failure line and the result both describe what was being trained, not an id."""
    messages = captured_log("ERROR")
    with patch.object(training_batch, "load_data_config", side_effect=ValueError("no config")):
        result = _run(
            tmp_path,
            training_presentation,
            assignment_id="exp-secret-9",
        )

    assert isinstance(result, AssignmentFailed)
    assert result.presentation is training_presentation
    assert messages == [f"{render_training_label(training_presentation)} failed: no config"]


def test_each_assignment_is_looked_up_under_its_own_derived_identity(
    run_assignment_dependencies: dict[str, MagicMock],
    tmp_path: Path,
    identity: StageIdentity,
    other_identity: StageIdentity,
    training_presentation: TrainingPresentation,
) -> None:
    """Two assignments derive and query their own identities, so a run for one
    can never satisfy the other."""
    run_assignment_dependencies["derive"].side_effect = [identity, other_identity]
    _run(
        tmp_path,
        training_presentation,
        assignment_id="train-job",
        tracking=run_assignment_dependencies["tracking"],
    )
    _run(
        tmp_path,
        training_presentation,
        assignment_id="search-job",
        tracking=run_assignment_dependencies["tracking"],
    )

    queried = [c.args[0] for c in run_assignment_dependencies["find"].call_args_list]
    assert queried == [identity, other_identity]
    assert run_assignment_dependencies["prepare_training_settings"].call_count == 2


@pytest.fixture
def real_mlflow(
    tmp_path: Path, mlflow_tracking_uri: str
) -> Iterator[tuple[dict[str, MagicMock], str]]:
    """Like run_assignment_dependencies, but the identity store hits a real
    sqlite-backed MLflow store — exercises the real tag round-trip end-to-end."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tracking_uri = mlflow_tracking_uri
    MlflowClient(tracking_uri=tracking_uri).create_experiment("Train")
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "resolve_dataset_artifacts") as resolve_dataset_artifacts,
        patch.object(training_batch, "training_identity") as derive,
        patch.object(training_batch, "prepare_training_settings") as prepare_training_settings,
    ):
        data_cfg = MagicMock()
        data_cfg.id = "assignment-dataset"
        data_cfg.output.data_dir = tmp_path
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"
        resolve_dataset_artifacts.return_value = MagicMock(
            rhs=MagicMock(path=data_dir),
            solutions=MagicMock(path=data_dir),
            matrix=MagicMock(path=data_dir),
        )
        prepare_training_settings.return_value = MagicMock(name="prepared-training")
        yield (
            {"derive": derive, "prepare_training_settings": prepare_training_settings},
            tracking_uri,
        )


def _seed_finished_run(client: MlflowClient, tmp_path: Path, identity: StageIdentity) -> str:
    experiment = client.get_experiment_by_name("Train")
    assert experiment is not None
    run = client.create_run(experiment.experiment_id, tags=identity.tags())
    checkpoint = tmp_path / "ckpt.ckpt"
    checkpoint.write_bytes(b"x")
    client.log_artifact(run.info.run_id, str(checkpoint), artifact_path="checkpoints")
    client.set_terminated(run.info.run_id, "FINISHED")
    return run.info.run_id


def test_real_store_reuses_run_with_the_same_identity(
    real_mlflow: tuple[dict[str, MagicMock], str],
    tmp_path: Path,
    identity: StageIdentity,
    training_presentation: TrainingPresentation,
) -> None:
    """A genuinely FINISHED, tagged, checkpointed run in a real MLflow store is reused."""
    deps, tracking_uri = real_mlflow
    deps["derive"].return_value = identity
    run_id = _seed_finished_run(MlflowClient(tracking_uri=tracking_uri), tmp_path, identity)

    result = _run(tmp_path, training_presentation, tracking_uri=tracking_uri)

    assert isinstance(result, AssignmentSucceeded)
    assert result.mlflow_run_id == run_id
    deps["prepare_training_settings"].assert_not_called()


def test_real_store_retrains_when_the_identity_changed(
    real_mlflow: tuple[dict[str, MagicMock], str],
    tmp_path: Path,
    identity: StageIdentity,
    other_identity: StageIdentity,
    training_presentation: TrainingPresentation,
) -> None:
    """A run keyed on other inputs (e.g. a since-regenerated dataset) is not reused."""
    deps, tracking_uri = real_mlflow
    _seed_finished_run(MlflowClient(tracking_uri=tracking_uri), tmp_path, identity)
    deps["derive"].return_value = other_identity

    result = _run(tmp_path, training_presentation, tracking_uri=tracking_uri)

    deps["prepare_training_settings"].assert_called_once()
    assert result is deps["prepare_training_settings"].return_value
