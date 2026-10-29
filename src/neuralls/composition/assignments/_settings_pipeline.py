"""Training job materialization pipeline for lower-case DLKit jobs."""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.assignments._job_types import TrainableJobConfig
from neuralls.composition.assignments.job_materializer import materialize_training_job
from neuralls.composition.assignments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.composition.generation.default_services import FS_SPLIT_STORE
from neuralls.platform.config.models.workspace import AssignmentWorkspace


def _configure_training_pipeline[T: TrainableJobConfig](
    settings: T,
    workspace: AssignmentWorkspace,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
    dataset_format: str,
) -> tuple[T, AssignmentWorkspace]:
    """Materialize one runnable training job through narrow patch stages."""
    return materialize_training_job(
        settings,
        workspace=workspace,
        features=features,
        targets=targets,
        contract=contract,
        dataset_format=dataset_format,
        split_store=FS_SPLIT_STORE,
    )
