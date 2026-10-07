"""Dataset assembly helpers for the training workflow.

Translates dataset storage artifacts into resolved DLKit data entries,
validates the runtime dataset contract, and leaves format-specific entry
construction to platform adapters.
"""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.job_config import FitJobConfig
from loguru import logger

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.storage.training_artifacts import (
    ArraySource,
    TrainingArrays,
    load_training_arrays,
)


def _validate_dataset_section(settings: AnyJobConfig) -> None:
    """Validate that the job config has a data section."""
    if settings.data is None:
        raise ValueError("Config is missing [data] section")


def _find_duplicate_entry_names(entries: tuple[DataEntry, ...]) -> set[str]:
    """Return duplicated non-null dataset entry names."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.name is None:
            continue
        if entry.name in seen:
            duplicates.add(entry.name)
        seen.add(entry.name)
    return duplicates


def validate_runtime_dataset_contract(
    settings: AnyJobConfig,
    contract: RuntimeDatasetContract,
) -> None:
    """Validate the local runtime dataset-entry contract for training."""
    _validate_dataset_section(settings)
    dataset = settings.data
    if dataset is None:
        raise ValueError("Config is missing [data] section")

    duplicate_features = _find_duplicate_entry_names(dataset.features)
    if duplicate_features:
        raise ValueError(
            f"Duplicate data feature entry names are not allowed: {sorted(duplicate_features)}"
        )

    duplicate_targets = _find_duplicate_entry_names(dataset.targets)
    if duplicate_targets:
        raise ValueError(
            f"Duplicate data target entry names are not allowed: {sorted(duplicate_targets)}"
        )

    target_names = {entry.name for entry in dataset.targets if entry.name is not None}
    unsupported_target_names = sorted(name for name in target_names if name != contract.target_name)
    if unsupported_target_names:
        raise ValueError(
            "data target placeholders must use only the resolved supervised "
            f"target name '{contract.target_name}', got {unsupported_target_names}."
        )

    if isinstance(settings, FitJobConfig) and _wants_matrix_feature(settings, contract):
        raise ValueError(
            f"A run.type='fit' job cannot declare a '{contract.matrix_input_name}' data "
            "feature — fit jobs are matrix-free by design (see PODCoarseningFittable's "
            "module docstring); the matrix is only ever available at comparison time."
        )

    training_cfg = getattr(settings, "training", None)
    loss_cfg = getattr(training_cfg, "loss", None) if training_cfg else None
    target_key = getattr(loss_cfg, "target_key", None)
    if target_key is None:
        return
    if target_key != contract.loss_target_key:
        raise ValueError(
            "training.loss.target_key must resolve to the runtime "
            f"supervised target '{contract.loss_target_key}', got '{target_key}'."
        )


def _primary_feature_name_from_settings(
    settings: AnyJobConfig,
    contract: RuntimeDatasetContract,
) -> str:
    """Resolve the dispatch name for the primary input feature."""
    dataset = settings.data
    if dataset is None:
        return contract.primary_input_name
    for entry in dataset.features:
        if entry.name is None or entry.name == contract.matrix_input_name:
            continue
        return entry.name
    return contract.primary_input_name


def _extra_feature_names_from_settings(
    settings: AnyJobConfig,
    contract: RuntimeDatasetContract,
) -> list[str]:
    """Extract extra feature names declared in ``[[data.features]]``."""
    dataset = settings.data
    if dataset is None:
        return []
    primary_name = _primary_feature_name_from_settings(settings, contract)
    base = {primary_name, contract.matrix_input_name}
    return [e.name for e in dataset.features if e.name is not None and e.name not in base]


def _wants_matrix_feature(
    settings: AnyJobConfig,
    contract: RuntimeDatasetContract,
) -> bool:
    """Whether the model opted into the system matrix via ``[[data.features]]``.

    DLKit never needs the matrix for its own data pipeline — it only ever reads
    ``rhs``/``solutions`` through its generic path-based entries. The matrix is
    exposed as a ``model_input=False`` context feature solely for matrix-aware
    architectures (e.g. ``GraphDataset``, see ``dataset.py::with_dataset_arrays``)
    that explicitly declare it by name; it must not be attached unconditionally.
    """
    dataset = settings.data
    if dataset is None:
        return False
    return any(entry.name == contract.matrix_input_name for entry in dataset.features)


def _create_feature_entries(
    arrays: TrainingArrays,
    contract: RuntimeDatasetContract,
    declared_extra_names: list[str],
    primary_name: str,
    *,
    wants_matrix_feature: bool,
) -> list[DataEntry]:
    """Create resolved feature entries from dataset artifacts."""
    if len(declared_extra_names) > len(arrays.parameter_sources):
        raise ValueError(
            f"Model declares {len(declared_extra_names)} extra features but dataset has "
            f"{len(arrays.parameter_sources)} parameters_* files."
        )
    base: list[DataEntry] = [
        _feature_entry_from_source(arrays.rhs_source, name=primary_name, model_input=True),
    ]
    if wants_matrix_feature:
        if arrays.matrix_format is not None:
            raise ValueError(
                f"Model declares a '{contract.matrix_input_name}' data feature, but this "
                f"dataset's matrix is stored as {arrays.matrix_format.value} — a group of "
                "sparse components, not a flat array DLKit's generic HDF5/zarr entry can "
                "open. Load it via the sparse_io backend and inject it as an in-memory "
                "feature instead (see dataset.py::with_dataset_arrays)."
            )
        base.append(
            _feature_entry_from_source(
                arrays.matrix_source, name=contract.matrix_input_name, model_input=False
            )
        )
    extras: list[DataEntry] = [
        _feature_entry_from_source(arrays.parameter_sources[i], name=name, model_input=True)
        for i, name in enumerate(declared_extra_names)
    ]
    return [*base, *extras]


def _create_target_entries(
    source: ArraySource,
    contract: RuntimeDatasetContract,
) -> list[DataEntry]:
    """Create resolved target entries from dataset artifacts."""
    return [_target_entry_from_source(source, name=contract.target_name)]


def resolve_dataset_format(arrays: TrainingArrays) -> str:
    """Return the on-disk format of the training arrays (e.g. "zarr", "npy", "hdf5")."""
    return arrays.rhs_source.path.suffix.lstrip(".") or "zarr"


def _load_and_prepare_data(
    settings: AnyJobConfig,
    workspace: AssignmentWorkspace,
    contract: RuntimeDatasetContract,
) -> tuple[TrainingArrays, list[DataEntry], list[DataEntry]]:
    """Resolve training data artifacts and build DLKit entries."""
    arrays = load_training_arrays(workspace.data_dir)
    fmt = resolve_dataset_format(arrays)
    logger.info(
        "Resolved training artifact sources ({} samples, format={}) from {}",
        arrays.sample_count,
        fmt,
        workspace.data_dir,
    )
    primary_name = _primary_feature_name_from_settings(settings, contract)
    extra_names = _extra_feature_names_from_settings(settings, contract)
    features = _create_feature_entries(
        arrays,
        contract,
        extra_names,
        primary_name,
        wants_matrix_feature=_wants_matrix_feature(settings, contract),
    )
    targets = _create_target_entries(arrays.solutions_source, contract)
    return arrays, features, targets


def _feature_entry_from_source(
    source: ArraySource,
    *,
    name: str,
    model_input: bool,
) -> DataEntry:
    """Build one feature entry from one persisted source."""
    return entry_from_path(
        source.path,
        name=name,
        model_input=model_input,
        role="feature",
        key=source.key,
    )


def _target_entry_from_source(
    source: ArraySource,
    *,
    name: str,
) -> DataEntry:
    """Build one target entry from one persisted source."""
    return entry_from_path(
        source.path,
        name=name,
        model_input=False,
        role="target",
        key=source.key,
    )
