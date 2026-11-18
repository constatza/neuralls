"""Tracking-only runtime patching for lower-case DLKit jobs."""

from __future__ import annotations

from neuralls.composition.assignments._job_types import DlkitJobConfig, TrainLikeJobConfig


def ensure_runtime_tracking(settings: DlkitJobConfig) -> DlkitJobConfig:
    """Ensure a tracking section exists so runtime wiring has one stable target."""
    if settings.tracking is not None:
        return settings
    return settings.patch({"tracking": {}})


def patch_training_tracking[T: TrainLikeJobConfig](
    settings: T,
    *,
    uri: str | None,
) -> T:
    """Enable MLflow tracking with the resolved tracking URI for training jobs."""
    return settings.patch({"tracking": {"backend": "mlflow", "uri": uri}})
