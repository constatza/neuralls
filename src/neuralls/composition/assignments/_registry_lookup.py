"""Shared registry-entry and config-path resolution for assignment workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neuralls.platform.config.loaders import missing_dataset_config_hint
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig, RegistryEntry
from neuralls.platform.config.registry import resolve_dataset_config_path, resolve_job_config_path


def _find_registry_entry(
    entries: Sequence[RegistryEntry],
    registry_id: str,
) -> RegistryEntry | None:
    """Look up one registry entry by id."""
    for entry in entries:
        if entry.id == registry_id:
            return entry
    return None


def _resolve_config_paths(
    assignment: AssignmentEntry,
    configs_dir: Path,
    cfg: CaseConfig,
) -> tuple[Path, Path]:
    """Resolve job and dataset config paths from an assignment entry.

    Args:
        assignment: Single ``[[assignments]]`` entry.
        configs_dir: Parent directory of the assignments TOML.
        cfg: Case config the assignment belongs to.

    Returns:
        Tuple of ``(job_config_path, data_config_path)``.

    Raises:
        FileNotFoundError: If either resolved config path does not exist.
    """
    job_path = resolve_job_config_path(
        cfg,
        configs_dir,
        assignment.job_id,
        assignment_id=assignment.id,
    )
    dataset_path = resolve_dataset_config_path(
        cfg,
        configs_dir,
        assignment.dataset_id,
        assignment_id=assignment.id,
    )
    if not job_path.exists():
        raise FileNotFoundError(f"Job config not found: {job_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_path}.{missing_dataset_config_hint(dataset_path)}"
        )
    return job_path, dataset_path
