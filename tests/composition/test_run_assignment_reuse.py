"""MLflow-backed reuse: an assignment only skips retraining if find_successful_run

reports a FINISHED run tagged with its own assignment_id — never based on a local
checkpoint or another assignment's run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralls.application.models import AssignmentResult
from neuralls.composition.assignments import training_batch
from neuralls.composition.assignments.training import PreparedTraining


@pytest.fixture
def run_assignment_dependencies(tmp_path: Path) -> Iterator[dict[str, MagicMock]]:
    """Patch every I/O dependency of ``run_assignment`` except the reuse decision."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "resolve_dataset_artifacts") as resolve_dataset_artifacts,
        patch.object(training_batch, "find_successful_run") as find_successful_run,
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
        yield {
            "find_successful_run": find_successful_run,
            "prepare_training_settings": prepare_training_settings,
        }


def _run(
    deps: dict[str, MagicMock],
    tmp_path: Path,
    *,
    assignment_id: str = "exp-1",
    force: bool = False,
) -> AssignmentResult | PreparedTraining:
    return training_batch.run_assignment(
        settings=MagicMock(),
        job_config_path=tmp_path / "job.toml",
        data_config_path=tmp_path / "data.toml",
        output_root=tmp_path,
        force=force,
        assignment_id=assignment_id,
        assignment_display_name="Assignment 1",
        mlflow_experiment_name="Train",
        tracking_uri="sqlite:///tracking.db",
    )


def test_train_skips_when_a_finished_run_already_exists(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """A FINISHED run tagged with this exact assignment_id short-circuits training."""
    run_assignment_dependencies["find_successful_run"].return_value = "existing-run-id"
    result = _run(run_assignment_dependencies, tmp_path)
    run_assignment_dependencies["prepare_training_settings"].assert_not_called()
    assert isinstance(result, AssignmentResult)
    assert result.status == "Success"


def test_train_runs_when_no_finished_run_exists(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """No matching FINISHED run means the assignment is prepared for the training sweep."""
    run_assignment_dependencies["find_successful_run"].return_value = None
    result = _run(run_assignment_dependencies, tmp_path)
    run_assignment_dependencies["prepare_training_settings"].assert_called_once()
    assert result is run_assignment_dependencies["prepare_training_settings"].return_value


def test_force_always_retrains_even_with_a_finished_run(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """force=True bypasses the MLflow reuse check entirely."""
    run_assignment_dependencies["find_successful_run"].return_value = "existing-run-id"
    result = _run(run_assignment_dependencies, tmp_path, force=True)
    run_assignment_dependencies["prepare_training_settings"].assert_called_once()
    run_assignment_dependencies["find_successful_run"].assert_not_called()
    assert result is run_assignment_dependencies["prepare_training_settings"].return_value


def test_run_assignment_returns_failed_result_on_unexpected_exception(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """An unexpected exception (e.g. dlkit's leaked-MLflow-run error on a search job)
    must be recorded as a Failed result, not raised — run_assignment_sweep's
    "failed assignments don't stop the batch" guarantee depends on this.
    """
    run_assignment_dependencies["find_successful_run"].return_value = None
    run_assignment_dependencies["prepare_training_settings"].side_effect = Exception(
        "Run with UUID abc123 is already active."
    )
    result = training_batch.run_assignment(
        settings=MagicMock(),
        job_config_path=tmp_path / "job.toml",
        data_config_path=tmp_path / "data.toml",
        output_root=tmp_path,
        force=False,
        assignment_id="search-job-2",
        assignment_display_name="Assignment 2",
        mlflow_experiment_name="Train",
        tracking_uri="sqlite:///tracking.db",
    )
    assert isinstance(result, AssignmentResult)
    assert result.status == "Failed"
    assert "already active" in (result.error or "")


def test_run_assignment_fails_cleanly_when_dataset_was_never_generated(
    tmp_path: Path,
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
            job_config_path=tmp_path / "job.toml",
            data_config_path=tmp_path / "data.toml",
            output_root=tmp_path,
            force=False,
            assignment_id="exp-1",
            assignment_display_name="Assignment 1",
            mlflow_experiment_name="Train",
            tracking_uri="sqlite:///tracking.db",
        )

    assert isinstance(result, AssignmentResult)
    assert result.status == "Failed"
    assert "generat" in (result.error or "").lower()
    prepare_training_settings.assert_not_called()


def test_two_assignments_never_share_a_lookup_key(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """A FINISHED run for one assignment_id must not skip a different assignment_id.

    find_successful_run is the sole reuse signal; querying it per-assignment_id
    (asserted below) is what keeps two assignments sharing an architecture+dataset
    from colliding on the same checkpoint slot.
    """
    run_assignment_dependencies["find_successful_run"].return_value = None
    _run(run_assignment_dependencies, tmp_path, assignment_id="train-job")
    _run(run_assignment_dependencies, tmp_path, assignment_id="search-job")

    queried_ids = [
        call.kwargs["assignment_id"]
        for call in run_assignment_dependencies["find_successful_run"].call_args_list
    ]
    assert queried_ids == ["train-job", "search-job"]
    assert run_assignment_dependencies["prepare_training_settings"].call_count == 2
