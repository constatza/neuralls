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

from neuralls.application.models import (
    AssignmentFailed,
    AssignmentSucceeded,
)
from neuralls.composition.assignments import training_batch
from neuralls.domain.identity import Reused
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.tracking.run_names import render_training_label
from tests.training_reuse_support import (
    IDENTITY,
    OTHER_IDENTITY,
    TRACKING_URI,
    patched_assignment_inputs,
    run_once,
)

type CapturedLog = Callable[[str], list[str]]


@pytest.fixture
def run_assignment_dependencies() -> Iterator[dict[str, MagicMock]]:
    """Patch every I/O dependency of ``run_assignment`` except the reuse decision."""
    tracking = MagicMock()
    store_cls = tracking.identity_store
    store_cls.return_value.find.return_value = None
    with patched_assignment_inputs(IDENTITY) as inputs:
        yield {
            "find": store_cls.return_value.find,
            "store_cls": store_cls,
            "tracking": tracking,
            "derive": inputs.derive,
            "prepare_training_settings": inputs.prepare_training_settings,
        }


def test_train_skips_when_an_identical_run_already_exists(
    run_assignment_dependencies: dict[str, MagicMock],
    training_presentation: TrainingPresentation,
) -> None:
    """A FINISHED run carrying this exact identity short-circuits training."""
    run_assignment_dependencies["find"].return_value = Reused(run_id="existing-run-id")
    result = run_once(
        training_presentation,
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )
    run_assignment_dependencies["prepare_training_settings"].assert_not_called()
    run_assignment_dependencies["find"].assert_called_once_with(IDENTITY)
    assert isinstance(result, AssignmentSucceeded)
    assert result.status == "Success"
    assert result.mlflow_run_id == "existing-run-id"


def test_train_runs_and_tags_the_identity_when_no_matching_run_exists(
    run_assignment_dependencies: dict[str, MagicMock],
    training_presentation: TrainingPresentation,
) -> None:
    """A miss prepares training and stamps the derived identity on the new run."""
    result = run_once(
        training_presentation,
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )
    prepare = run_assignment_dependencies["prepare_training_settings"]
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["extra_tags"] == IDENTITY.tags()
    assert result is prepare.return_value


def test_force_always_retrains_even_with_a_matching_run(
    run_assignment_dependencies: dict[str, MagicMock],
    training_presentation: TrainingPresentation,
) -> None:
    """force=True bypasses the reuse check entirely."""
    run_assignment_dependencies["find"].return_value = Reused(run_id="existing-run-id")
    result = run_once(
        training_presentation,
        force=True,
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )
    run_assignment_dependencies["prepare_training_settings"].assert_called_once()
    run_assignment_dependencies["find"].assert_not_called()
    assert result is run_assignment_dependencies["prepare_training_settings"].return_value


def test_run_assignment_returns_failed_result_on_unexpected_exception(
    run_assignment_dependencies: dict[str, MagicMock],
    training_presentation: TrainingPresentation,
) -> None:
    """An unexpected exception (e.g. dlkit's leaked-MLflow-run error on a search job)
    must be recorded as a Failed result, not raised — run_assignment_sweep's
    "failed assignments don't stop the batch" guarantee depends on this.
    """
    run_assignment_dependencies["prepare_training_settings"].side_effect = Exception(
        "Run with UUID abc123 is already active."
    )
    result = run_once(
        training_presentation,
        assignment_id="search-job-2",
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )
    assert isinstance(result, AssignmentFailed)
    assert result.status == "Failed"
    assert result.error is not None


def test_run_assignment_fails_cleanly_when_dataset_was_never_generated(
    training_presentation: TrainingPresentation,
) -> None:
    """Training must never auto-generate a missing dataset — that's the generate
    stage's exclusive job. If nobody ran `generate` first, run_assignment must
    fail cleanly (not silently regenerate) with an error pointing at the missing
    dataset, and must never reach training preparation.
    """
    absent = MagicMock(spec=Path)
    absent.exists.return_value = False
    absent.__str__.return_value = str(Path("workspace/data/never-generated/missing"))
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(
            training_batch,
            "resolve_dataset_artifacts",
            return_value=MagicMock(
                rhs=MagicMock(path=absent),
                solutions=MagicMock(path=absent),
                matrix=MagicMock(path=absent),
            ),
        ),
        patch.object(training_batch, "prepare_training_settings") as prepare_training_settings,
    ):
        data_cfg = MagicMock()
        data_cfg.id = "never-generated"
        data_cfg.output.data_dir = Path("workspace/data")
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"

        result = run_once(training_presentation, tracking=MagicMock(), tracking_uri=TRACKING_URI)

    assert isinstance(result, AssignmentFailed)
    assert result.error == (
        f"Required data files not found in {Path('workspace/data/never-generated')}:\n"
        + "\n".join([f"  - {absent}"] * 3)
    )
    prepare_training_settings.assert_not_called()


def test_a_failed_assignment_is_logged_by_its_training_presentation(
    training_presentation: TrainingPresentation, captured_log: CapturedLog
) -> None:
    """The failure line and the result both describe what was being trained, not an id."""
    messages = captured_log("ERROR")
    with patch.object(training_batch, "load_data_config", side_effect=ValueError("no config")):
        result = run_once(
            training_presentation,
            assignment_id="exp-secret-9",
            tracking=MagicMock(),
            tracking_uri=TRACKING_URI,
        )

    assert isinstance(result, AssignmentFailed)
    assert result.presentation is training_presentation
    assert messages == [f"{render_training_label(training_presentation)} failed: no config"]


def test_each_assignment_is_looked_up_under_its_own_derived_identity(
    run_assignment_dependencies: dict[str, MagicMock],
    training_presentation: TrainingPresentation,
) -> None:
    """Two assignments derive and query their own identities, so a run for one
    can never satisfy the other."""
    run_assignment_dependencies["derive"].side_effect = [IDENTITY, OTHER_IDENTITY]
    run_once(
        training_presentation,
        assignment_id="train-job",
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )
    run_once(
        training_presentation,
        assignment_id="search-job",
        tracking=run_assignment_dependencies["tracking"],
        tracking_uri=TRACKING_URI,
    )

    queried = [c.args[0] for c in run_assignment_dependencies["find"].call_args_list]
    assert queried == [IDENTITY, OTHER_IDENTITY]
    assert run_assignment_dependencies["prepare_training_settings"].call_count == 2
