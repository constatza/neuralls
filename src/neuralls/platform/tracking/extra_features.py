"""Utilities for storing and retrieving extra feature names from MLflow run tags."""

from __future__ import annotations

from collections.abc import Iterable

from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.mlflow import quote_filter_value

EXTRA_FEATURE_NAMES_TAG: str = "neuralls.extra_feature_names"


def fetch_extra_feature_names(run_id: str, *, client: MlflowClient) -> tuple[str, ...]:
    """Fetch extra feature names logged during training for a given MLflow run.

    Reads the ``neuralls.extra_feature_names`` tag from the run, splits on
    comma, filters empty strings, and returns the result as a tuple.

    Args:
        run_id: MLflow run ID from the training run.
        client: Configured ``MlflowClient`` pointing at the tracking server.

    Returns:
        Tuple of extra feature names, or empty tuple if the tag is absent or empty.
    """
    run = client.get_run(run_id)
    raw = run.data.tags.get(EXTRA_FEATURE_NAMES_TAG, "")
    return tuple(name for name in raw.split(",") if name)


def log_extra_feature_names_tag(
    tracking_uri: str,
    run_id: str,
    extra_names: Iterable[str],
) -> None:
    """Log extra feature names as a run tag on an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run ID to tag.
        extra_names: Set of extra feature names declared in the model TOML.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    tag_value = ",".join(extra_names)
    client.set_tag(run_id, EXTRA_FEATURE_NAMES_TAG, tag_value)


def _all_experiment_ids(client: MlflowClient) -> list[str]:
    """Return every experiment id visible to this client."""
    return [experiment.experiment_id for experiment in client.search_experiments()]


def _lookup_run_id_for_model(entry_id: str, client: MlflowClient) -> str | None:
    """Look up the most recently started training run tagged for an assignment entry.

    Registration is no longer automatic, so there is generally no registered
    model to look up. Instead, this searches raw MLflow runs across all
    experiments for the most recent run tagged with ``assignment_id ==
    entry_id`` — the same tag ``composition/tracking/run_specs.py``'s
    ``build_training_run_spec`` sets on every training run.

    Args:
        entry_id: Assignment registry ID matched against the run's
            ``assignment_id`` tag.
        client: Configured MLflow client.

    Returns:
        MLflow run ID of the most recently started matching run, or None if
        no run carries that tag.
    """
    experiment_ids = _all_experiment_ids(client)
    if not experiment_ids:
        return None
    filter_string = f"tags.`assignment_id` = '{quote_filter_value(entry_id)}'"
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def fetch_extra_input_names_for_model(entry_id: str, client: MlflowClient) -> tuple[str, ...]:
    """Fetch extra input names from the training run tag for an assignment entry.

    Looks up the most recently tagged training run for ``entry_id``, then
    reads the ``neuralls.extra_feature_names`` tag from that run.

    Args:
        entry_id: Assignment registry ID matched against the run's
            ``assignment_id`` tag.
        client: Configured MLflow client.

    Returns:
        Tuple of extra input names, or empty tuple if unavailable.
    """
    run_id = _lookup_run_id_for_model(entry_id, client)
    if run_id is None:
        return ()
    return fetch_extra_feature_names(run_id, client=client)


__all__ = [
    "EXTRA_FEATURE_NAMES_TAG",
    "fetch_extra_feature_names",
    "fetch_extra_input_names_for_model",
    "log_extra_feature_names_tag",
]
