"""Tests for find_successful_run's filter-string construction and result parsing,
find_successful_comparison_run's equivalent for comparisons, and mark_run_failed's
best-effort status update."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralls.platform.tracking.mlflow_client import (
    find_successful_comparison_run,
    find_successful_run,
    mark_run_failed,
)


def _mock_client(experiment_id: str | None, runs: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    if experiment_id is None:
        client.get_experiment_by_name.return_value = None
    else:
        client.get_experiment_by_name.return_value = MagicMock(experiment_id=experiment_id)
    client.search_runs.return_value = runs
    return client


def test_returns_none_when_experiment_does_not_exist() -> None:
    """No MLflow experiment bucket means nothing to search — return None."""
    client = _mock_client(experiment_id=None, runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result is None
    client.search_runs.assert_not_called()


def test_returns_none_when_no_matching_run_found() -> None:
    """An existing experiment with no matching run also returns None."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result is None


def test_returns_run_id_of_most_recent_match() -> None:
    """The first run in the (already start_time-DESC-ordered) result is returned,
    once it's confirmed to actually have a checkpoint artifact."""
    run = MagicMock()
    run.info.run_id = "run-abc"
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[run])
    with (
        patch("mlflow.tracking.MlflowClient", return_value=client),
        patch("dlkit.mlflow.has_checkpoint_artifact", return_value=True),
    ):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result == "run-abc"
    _, kwargs = client.search_runs.call_args
    assert kwargs["max_results"] == 20


def test_falls_through_to_older_run_when_newest_has_no_checkpoint() -> None:
    """When the newest FINISHED run lacks a checkpoint, the next-newest one with a
    checkpoint is returned instead — a broken newest attempt must not shadow an
    older, genuinely complete run for the same assignment."""
    newest_run = MagicMock()
    newest_run.info.run_id = "run-newest"
    older_run = MagicMock()
    older_run.info.run_id = "run-older"
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[newest_run, older_run])

    def _has_checkpoint(run_id: str, *, tracking_uri: str | None = None) -> bool:
        return run_id == "run-older"

    with (
        patch("mlflow.tracking.MlflowClient", return_value=client),
        patch("dlkit.mlflow.has_checkpoint_artifact", side_effect=_has_checkpoint),
    ):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result == "run-older"


def test_returns_none_when_no_candidate_has_a_checkpoint() -> None:
    """No FINISHED candidate has a checkpoint artifact — return None rather than
    resolving to a broken run."""
    run_one = MagicMock()
    run_one.info.run_id = "run-one"
    run_two = MagicMock()
    run_two.info.run_id = "run-two"
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[run_one, run_two])
    with (
        patch("mlflow.tracking.MlflowClient", return_value=client),
        patch("dlkit.mlflow.has_checkpoint_artifact", return_value=False),
    ):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result is None


def test_filter_string_scopes_to_assignment_id_and_finished_status() -> None:
    """The search filter is scoped to this exact assignment_id and FINISHED status —
    the two properties that make the reuse check safe across different assignments
    and against crashed/still-running prior attempts."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="my-assignment",
        )
    _, kwargs = client.search_runs.call_args
    assert kwargs["experiment_ids"] == ["mlflow-exp-1"]
    assert "tags.assignment_id = 'my-assignment'" in kwargs["filter_string"]
    assert "attributes.status = 'FINISHED'" in kwargs["filter_string"]


def test_two_different_assignment_ids_produce_different_filters() -> None:
    """Distinct assignment_ids never collide on the same search filter."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="train-job",
        )
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="search-job",
        )
    first_filter = client.search_runs.call_args_list[0].kwargs["filter_string"]
    second_filter = client.search_runs.call_args_list[1].kwargs["filter_string"]
    assert "train-job" in first_filter
    assert "search-job" in second_filter
    assert first_filter != second_filter


def test_dataset_hash_is_included_in_filter_when_given() -> None:
    """A dataset_hash argument adds a tag filter so a regenerated dataset (whose
    hash no longer matches) doesn't read as an already-trained run."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
            dataset_hash="abc123",
        )
    _, kwargs = client.search_runs.call_args
    assert "tags.dataset_hash = 'abc123'" in kwargs["filter_string"]


def test_dataset_hash_omitted_when_not_given() -> None:
    """Omitting dataset_hash preserves the original assignment_id-only filter."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    _, kwargs = client.search_runs.call_args
    assert "dataset_hash" not in kwargs["filter_string"]


def _mock_comparison_client(experiment_id: str | None, runs: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    if experiment_id is None:
        client.get_experiment_by_name.return_value = None
    else:
        client.get_experiment_by_name.return_value = MagicMock(experiment_id=experiment_id)
    client.search_runs.return_value = runs
    return client


def test_find_successful_comparison_run_returns_none_when_experiment_missing() -> None:
    client = _mock_comparison_client(experiment_id=None, runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_comparison_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Compare",
            comparison_id="cmp-1",
            checkpoint_dependency_hash="sig-1",
        )
    assert result is None


def test_find_successful_comparison_run_returns_none_when_no_match() -> None:
    client = _mock_comparison_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_comparison_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Compare",
            comparison_id="cmp-1",
            checkpoint_dependency_hash="sig-1",
        )
    assert result is None


def test_find_successful_comparison_run_returns_matching_run_id() -> None:
    run = MagicMock()
    run.info.run_id = "comp-run-1"
    client = _mock_comparison_client(experiment_id="mlflow-exp-1", runs=[run])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_comparison_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Compare",
            comparison_id="cmp-1",
            checkpoint_dependency_hash="sig-1",
        )
    assert result == "comp-run-1"
    _, kwargs = client.search_runs.call_args
    assert "tags.comparison_id = 'cmp-1'" in kwargs["filter_string"]
    assert "tags.checkpoint_dependency_hash = 'sig-1'" in kwargs["filter_string"]
    assert "attributes.status = 'FINISHED'" in kwargs["filter_string"]


def test_find_successful_comparison_run_distinguishes_signatures() -> None:
    """A changed checkpoint_dependency_hash (e.g. after retraining) must not match
    a run logged under the old signature."""
    client = _mock_comparison_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_comparison_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Compare",
            comparison_id="cmp-1",
            checkpoint_dependency_hash="sig-new",
        )
    _, kwargs = client.search_runs.call_args
    assert "sig-new" in kwargs["filter_string"]


def test_mark_run_failed_calls_set_terminated_with_failed_status() -> None:
    """mark_run_failed terminates the given run with status FAILED."""
    client = MagicMock()
    with patch("mlflow.tracking.MlflowClient", return_value=client) as mlflow_client_cls:
        mark_run_failed(run_id="run-abc", tracking_uri="sqlite:///tracking.db")
    mlflow_client_cls.assert_called_once_with(tracking_uri="sqlite:///tracking.db")
    client.set_terminated.assert_called_once_with("run-abc", status="FAILED")


def test_mark_run_failed_swallows_exceptions_from_set_terminated() -> None:
    """mark_run_failed is best-effort — an exception from set_terminated must never
    propagate, since this is called from inside an except block and must not mask
    the original exception."""
    client = MagicMock()
    client.set_terminated.side_effect = RuntimeError("tracking server unreachable")
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        mark_run_failed(run_id="run-abc", tracking_uri="sqlite:///tracking.db")
