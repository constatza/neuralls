"""Case-config registry resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    RegistryEntry,
)
from neuralls.platform.config.resolution import resolve_registry_path
from neuralls.shared.scientific_description.models import ValidatedLabel


@dataclass(frozen=True, slots=True)
class ResolvedAssignmentBinding:
    """Resolved assignment registry entry with concrete config paths."""

    assignment_id: str
    dataset_id: str
    job_id: str
    data_config_path: Path
    job_config_path: Path
    checkpoint_path: Path | None = None
    architecture: ValidatedLabel | None = None
    """The job's explicit neural architecture name; never derived from its training title."""


def _lookup_registry_entry[Entry: RegistryEntry](
    entries: Sequence[Entry],
    registry_id: str,
) -> Entry | None:
    """Find one registry entry by id."""
    for entry in entries:
        if entry.id == registry_id:
            return entry
    return None


def _require_registry_entry[Entry: RegistryEntry](
    entries: Sequence[Entry],
    *,
    registry_id: str,
    registry_section: str,
    owner_kind: str | None = None,
    owner_id: str | None = None,
) -> Entry:
    """Return a registry entry or raise a focused validation error."""
    entry = _lookup_registry_entry(entries, registry_id)
    if entry is not None:
        return entry
    if owner_kind is not None and owner_id is not None:
        raise ValueError(
            f"{owner_kind} '{owner_id}' references {registry_section[:-1]} id "
            f"'{registry_id}', but [[{registry_section}]] does not define it."
        )
    raise ValueError(
        f"{registry_section[:-1].capitalize()} id '{registry_id}' is not defined "
        f"in [[{registry_section}]]."
    )


def resolve_dataset_config_path(
    cfg: CaseConfig,
    config_dir: Path,
    dataset_id: str,
    *,
    assignment_id: str | None = None,
) -> Path:
    """Resolve a dataset config path from the case registry."""
    entry = _require_registry_entry(
        cfg.datasets,
        registry_id=dataset_id,
        registry_section="datasets",
        owner_kind="Assignment" if assignment_id is not None else None,
        owner_id=assignment_id,
    )
    return resolve_registry_path(config_dir, entry.path)


def resolve_job_config_path(
    cfg: CaseConfig,
    config_dir: Path,
    job_id: str,
    *,
    assignment_id: str | None = None,
) -> Path:
    """Resolve a job config path from the case registry."""
    entry = _require_registry_entry(
        cfg.jobs,
        registry_id=job_id,
        registry_section="jobs",
        owner_kind="Assignment" if assignment_id is not None else None,
        owner_id=assignment_id,
    )
    return resolve_registry_path(config_dir, entry.path)


def resolve_assignment_binding(
    cfg: CaseConfig,
    config_dir: Path,
    entry: AssignmentEntry,
) -> ResolvedAssignmentBinding:
    """Resolve a single assignment entry into concrete config paths."""
    _require_registry_entry(
        cfg.datasets,
        registry_id=entry.dataset_id,
        registry_section="datasets",
        owner_kind="Assignment",
        owner_id=entry.id,
    )
    job_entry = _require_registry_entry(
        cfg.jobs,
        registry_id=entry.job_id,
        registry_section="jobs",
        owner_kind="Assignment",
        owner_id=entry.id,
    )
    return ResolvedAssignmentBinding(
        assignment_id=entry.id,
        dataset_id=entry.dataset_id,
        job_id=entry.job_id,
        data_config_path=resolve_dataset_config_path(
            cfg,
            config_dir,
            entry.dataset_id,
            assignment_id=entry.id,
        ),
        job_config_path=resolve_job_config_path(
            cfg,
            config_dir,
            entry.job_id,
            assignment_id=entry.id,
        ),
        checkpoint_path=(
            resolve_registry_path(config_dir, entry.checkpoint_path)
            if entry.checkpoint_path is not None
            else None
        ),
        architecture=job_entry.architecture_label,
    )


def get_assignment_binding(
    cfg: CaseConfig,
    config_dir: Path,
    assignment_id: str,
) -> ResolvedAssignmentBinding:
    """Resolve one assignment by id."""
    for entry in cfg.assignments:
        if entry.id == assignment_id:
            return resolve_assignment_binding(cfg, config_dir, entry)
    raise KeyError(f"Assignment '{assignment_id}' not found in case config registry.")


def list_assignment_bindings(
    cfg: CaseConfig,
    config_dir: Path,
) -> list[ResolvedAssignmentBinding]:
    """Resolve all assignment entries with concrete config paths."""
    return [resolve_assignment_binding(cfg, config_dir, entry) for entry in cfg.assignments]
