"""Utilities for storing and retrieving extra feature names from MLflow run tags."""

from __future__ import annotations

from collections.abc import Iterable

from mlflow.tracking import MlflowClient

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


__all__ = [
    "EXTRA_FEATURE_NAMES_TAG",
    "fetch_extra_feature_names",
    "log_extra_feature_names_tag",
]
