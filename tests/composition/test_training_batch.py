"""Regression tests for run_assignment_sweep's dlkit LifecycleHooks wiring.

Covers two things the sweep must get right for every multirun child,
regardless of job kind (`train`/`search`/`fit` alike — dlkit hardcodes
`duration_seconds=0.0` for every executor, so nothing here can rely on dlkit
having measured its own duration):

1. Device memory is released after every child, success or failure — dlkit
   runs every sweep child sequentially in-process with no allocator reset of
   its own (see composition/README.md), so a failed or CUDA-heavy child
   otherwise starves every child that runs after it.
2. Each child's wall time/peak memory is measured from the outside (bridging
   `on_child_planned` at dispatch to `on_child_completed`/`on_child_failed`
   at the end) and logged onto that child's own MLflow run, since dlkit never
   logs a real duration itself.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from dlkit.common import ChildFailure, ChildPlannedEvent, ChildSuccess, LifecycleHooks

from neuralls.composition.assignments.training import PreparedTraining
from neuralls.composition.assignments.training_batch import (
    _ChildTimingTracker,
    run_assignment_sweep,
)
from neuralls.shared.constants import (
    TRAINING_CHILD_DURATION_METRIC_KEY,
    TRAINING_CHILD_PEAK_MEMORY_METRIC_KEY,
)
from neuralls.shared.device import ResourceUsage, ResourceUsageToken


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


def test_sweep_registers_child_timing_tracker_as_lifecycle_hooks(tmp_path: Path) -> None:
    """`run_multirun_spec` must receive planned/completed/failed hooks bound to
    one `_ChildTimingTracker` instance — otherwise device memory is never
    released after a child, and no child's duration ever reaches MLflow.
    """
    child_failure = MagicMock(spec=ChildFailure)
    child_failure.message = "boom"
    with ExitStack() as stack:
        run_multirun_spec = _patch_sweep_collaborators(stack, child_outcome=child_failure)
        run_assignment_sweep(tmp_path / "case.toml")

    hooks = run_multirun_spec.call_args.kwargs["hooks"]
    assert isinstance(hooks, LifecycleHooks)
    tracker = hooks.on_child_completed.__self__  # type: ignore[attr-defined]
    assert isinstance(tracker, _ChildTimingTracker)
    assert hooks.on_child_planned == tracker.on_planned
    assert hooks.on_child_completed == tracker.on_finished
    assert hooks.on_child_failed == tracker.on_finished


class TestChildTimingTracker:
    """Unit tests for `_ChildTimingTracker`, independent of the sweep wiring."""

    def _tracker(self) -> _ChildTimingTracker:
        return _ChildTimingTracker(tracking_uri="tracking-uri")

    def _patched(self, stack: ExitStack, *, usage: ResourceUsage) -> dict[str, MagicMock]:
        mocks = {
            "resolve_device": stack.enter_context(
                patch("neuralls.composition.assignments.training_batch.resolve_device")
            ),
            "begin": stack.enter_context(
                patch(
                    "neuralls.composition.assignments.training_batch.begin_resource_usage",
                    return_value=ResourceUsageToken(device=MagicMock(), start=0.0, rss_before=0),
                )
            ),
            "end": stack.enter_context(
                patch(
                    "neuralls.composition.assignments.training_batch.end_resource_usage",
                    return_value=usage,
                )
            ),
            "log_metric": stack.enter_context(
                patch("neuralls.composition.assignments.training_batch.log_metric_to_run")
            ),
            "release": stack.enter_context(
                patch("neuralls.composition.assignments.training_batch.release_device_memory")
            ),
        }
        return mocks

    def test_on_finished_logs_duration_and_memory_for_a_planned_success(self) -> None:
        """Fit-kind and train-kind children are handled identically — the tracker
        only ever looks at `child_id`/`run_id`, never a job-kind label.
        """
        usage = ResourceUsage(wall_time_seconds=1.5, peak_memory_bytes=2048)
        for kind in ("fit", "train"):
            tracker = self._tracker()
            with ExitStack() as stack:
                mocks = self._patched(stack, usage=usage)
                tracker.on_planned(ChildPlannedEvent(
                    child_id=f"{kind}-child", label=kind, run_name="r", tags={}, params={}, metadata={},
                ))  # fmt: skip
                outcome = MagicMock(spec=ChildSuccess)
                outcome.child_id = f"{kind}-child"
                outcome.run_id = "run-123"
                tracker.on_finished(outcome)

            mocks["end"].assert_called_once()
            mocks["log_metric"].assert_any_call(
                "run-123", TRAINING_CHILD_DURATION_METRIC_KEY, 1.5, "tracking-uri"
            )
            mocks["log_metric"].assert_any_call(
                "run-123", TRAINING_CHILD_PEAK_MEMORY_METRIC_KEY, 2048, "tracking-uri"
            )
            mocks["release"].assert_called_once()

    def test_on_finished_skips_logging_without_a_matching_planned_entry(self) -> None:
        """No `on_child_planned` was ever seen for this child_id — logging is a
        no-op, not an error, but memory release still always happens.
        """
        tracker = self._tracker()
        with ExitStack() as stack:
            mocks = self._patched(stack, usage=ResourceUsage(0.0, 0))
            outcome = MagicMock(spec=ChildFailure)
            outcome.child_id = "never-planned"
            outcome.run_id = "run-123"
            tracker.on_finished(outcome)

        mocks["end"].assert_not_called()
        mocks["log_metric"].assert_not_called()
        mocks["release"].assert_called_once()

    def test_on_finished_skips_logging_when_run_id_is_none(self) -> None:
        """A child outcome with no MLflow run at all has nothing to log onto."""
        tracker = self._tracker()
        with ExitStack() as stack:
            mocks = self._patched(stack, usage=ResourceUsage(1.0, 1))
            tracker.on_planned(ChildPlannedEvent(
                child_id="c", label="c", run_name="r", tags={}, params={}, metadata={},
            ))  # fmt: skip
            outcome = MagicMock(spec=ChildFailure)
            outcome.child_id = "c"
            outcome.run_id = None
            tracker.on_finished(outcome)

        mocks["log_metric"].assert_not_called()
        mocks["release"].assert_called_once()

    def test_on_finished_warns_and_continues_when_logging_fails(self) -> None:
        """A metrics-logging failure (no tracking server, etc.) must never
        propagate — it would turn a real training success/failure into a
        worse, unrelated crash.
        """
        tracker = self._tracker()
        with ExitStack() as stack:
            mocks = self._patched(stack, usage=ResourceUsage(1.0, 1))
            mocks["log_metric"].side_effect = RuntimeError("no tracking server")
            tracker.on_planned(ChildPlannedEvent(
                child_id="c", label="c", run_name="r", tags={}, params={}, metadata={},
            ))  # fmt: skip
            outcome = MagicMock(spec=ChildSuccess)
            outcome.child_id = "c"
            outcome.run_id = "run-123"
            tracker.on_finished(outcome)  # must not raise

        mocks["release"].assert_called_once()
