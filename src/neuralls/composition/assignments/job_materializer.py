"""Compose narrow runtime patch stages into runnable DLKit jobs."""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.assignments._job_types import AnyJobConfig, TrainableJobConfig
from neuralls.composition.assignments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.composition.assignments.runtime_dataset_patcher import patch_runtime_dataset
from neuralls.composition.assignments.runtime_split_patcher import patch_split_settings
from neuralls.composition.assignments.runtime_tracking_patcher import ensure_runtime_tracking
from neuralls.composition.assignments.runtime_workspace_patcher import (
    patch_dataloader_runtime,
    patch_runtime_workspace_for_job,
)
from neuralls.platform.config.models.workspace import AssignmentWorkspace


def materialize_training_job[T: TrainableJobConfig](
    settings: T,
    *,
    workspace: AssignmentWorkspace,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
    dataset_format: str,
) -> tuple[T, AssignmentWorkspace]:
    """Apply runtime dataset, split-pinning, and workspace stages in order.

    Tracking is already fully resolved by ``load_assignment()`` before this
    runs; re-patching it here would silently drop the resolved URI. Split
    pinning runs right after the dataset is wired in, before dataloader/
    workspace patching, so every job kind trains, searches, fits, or
    evaluates against the same persisted held-out partition for this
    dataset. Workspace patching is a no-op for job kinds with no trainer
    (``FitJobConfig``) — see ``patch_runtime_workspace_for_job``.
    """
    settings = patch_runtime_dataset(
        settings,
        features=features,
        targets=targets,
        contract=contract,
    )
    settings = patch_split_settings(settings, dataset_dir=workspace.data_dir)
    settings = patch_dataloader_runtime(settings, dataset_format=dataset_format)
    settings = patch_runtime_workspace_for_job(
        settings,
        output_dir=workspace.root_dir,
    )
    return settings, workspace


def materialize_inference_job(settings: AnyJobConfig) -> AnyJobConfig:
    """Apply inference-safe runtime tracking normalization only."""
    return ensure_runtime_tracking(settings)
