"""Regression test for run_assignment_sweep's dlkit LifecycleHooks wiring.

Doesn't (and can't, without a GPU) prove memory is actually reclaimed —
`release_device_memory()` is a one-line `torch.cuda.empty_cache()` guard,
not worth testing itself. What regresses silently is the *wiring*: dlkit
runs every sweep child sequentially in-process with no allocator reset of
its own (see composition/README.md), so `run_assignment_sweep` must hand it
a `LifecycleHooks` that releases device memory after every child, success or
failure. This locks that wiring down without touching CUDA.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from dlkit.common import ChildFailure, LifecycleHooks

from neuralls.composition.assignments.training import PreparedTraining
from neuralls.composition.assignments.training_batch import (
    _release_device_memory_on_child_outcome,
    run_assignment_sweep,
)


def _patch_sweep_collaborators(stack: ExitStack, *, child_outcome: MagicMock) -> MagicMock:
    """Patch every collaborator `run_assignment_sweep` needs to reach `run_multirun_spec`.

    Returns the `run_multirun_spec` mock so the test can inspect how it was called.
    """
    assignment = MagicMock(spec=["spec", "workspace"])
    assignment.spec.assignment_id = "assignment-1"
    prepared = MagicMock(spec=PreparedTraining)
    prepared.resolved_assignment_display_name = "assignment-1"
    child_outcome.child_id = "assignment-1"

    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.require_settings",
            side_effect=lambda settings, **_: settings,
        )
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.load_assignment_batch",
            return_value=MagicMock(assignments=(assignment,), output_root=Path("/out")),
        )
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.load_validated_case_config",
            return_value=(MagicMock(), MagicMock()),
        )
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.build_workflow_environment",
            return_value=MagicMock(tracking_uri="tracking-uri"),
        )
    )
    stack.enter_context(
        patch("neuralls.composition.assignments.training_batch.scoped_mlflow_environment")
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.build_session_run_spec",
            return_value=("session-name", MagicMock(as_mlflow_tags=dict)),
        )
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.run_assignment",
            return_value=prepared,
        )
    )
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.to_run_spec",
            return_value=MagicMock(),
        )
    )
    stack.enter_context(
        patch("neuralls.composition.assignments.training_batch.cleanup_prepared_training")
    )
    stack.enter_context(
        patch("neuralls.composition.assignments.training_batch.finalize_session_parent_run")
    )
    run_multirun_spec = stack.enter_context(
        patch(
            "neuralls.composition.assignments.training_batch.run_multirun_spec",
            return_value=MagicMock(
                children=(child_outcome,), tracking_uri="tracking-uri", parent_run_id="parent-1"
            ),
        )
    )
    return run_multirun_spec


def test_sweep_registers_device_memory_release_as_child_lifecycle_hooks(
    tmp_path: Path,
) -> None:
    """`run_multirun_spec` must receive hooks that release CUDA memory after
    every child — otherwise a failed fit-job child's allocated-but-orphaned
    CUDA blocks starve every child that runs after it in the same process.
    """
    child_failure = MagicMock(spec=ChildFailure)
    child_failure.message = "boom"
    with ExitStack() as stack:
        run_multirun_spec = _patch_sweep_collaborators(stack, child_outcome=child_failure)
        run_assignment_sweep(tmp_path / "case.toml")

    hooks = run_multirun_spec.call_args.kwargs["hooks"]
    assert isinstance(hooks, LifecycleHooks)
    assert hooks.on_child_completed is _release_device_memory_on_child_outcome
    assert hooks.on_child_failed is _release_device_memory_on_child_outcome


def test_child_lifecycle_hook_releases_device_memory() -> None:
    """The registered callback itself must actually call `release_device_memory`."""
    with patch("neuralls.composition.assignments.training_batch.release_device_memory") as release:
        _release_device_memory_on_child_outcome(MagicMock())

    release.assert_called_once()
