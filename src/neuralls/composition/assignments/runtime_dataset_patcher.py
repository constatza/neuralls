"""Dataset-only runtime patching for lower-case DLKit jobs."""

from __future__ import annotations

from dlkit.config import DataSettings
from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.assignments._dataset_assembly import validate_runtime_dataset_contract
from neuralls.composition.assignments._job_types import DlkitJobConfig
from neuralls.composition.assignments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.config.dataset_entries import apply_placeholder_metadata


def patch_runtime_dataset[T: DlkitJobConfig](
    settings: T,
    *,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
) -> T:
    """Inject resolved runtime dataset entries into ``settings.data`` only."""
    validate_runtime_dataset_contract(settings, contract)
    base_data = settings.data or DataSettings()
    feature_entries = apply_placeholder_metadata(features, base_data.features)
    target_entries = apply_placeholder_metadata(targets, base_data.targets)
    return settings.patch(
        {
            "data": {
                "features": feature_entries,
                "targets": target_entries,
                "name": base_data.name or "FlexibleDataset",
            }
        }
    )
