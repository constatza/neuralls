"""Training-stage identity: everything that determines a trained model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neuralls.domain.identity import IdentityTag, StageIdentity
from neuralls.platform.config.dlkit_bridge import load_job_config
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.dataset_digest import current_dataset_digest
from neuralls.shared.digest import canonical_digest

# DLKit job sections that only say where/how results are reported, not what is trained.
# Every other section — including ones added by future DLKit versions — is hashed.
_NON_IDENTITY_JOB_SECTIONS = frozenset({"tracking", "plots", "experiment"})


def assignment_dataset_dir(data_config_path: Path, settings: NeurallsSettings) -> Path:
    """Resolve the directory holding an assignment's generated dataset.

    Single definition shared by training and comparison, so both look at the
    same directory.

    Args:
        data_config_path (Path): Dataset TOML the assignment is bound to.
        settings (NeurallsSettings): Runtime settings (default data root).

    Returns:
        Path: ``<data_dir>/<dataset id>``.
    """
    data_cfg = load_data_config(data_config_path, settings)
    dataset_id = resolve_dataset_identity(data_cfg=data_cfg, config_path=data_config_path).name
    return (data_cfg.output.data_dir or settings.processed_dir) / dataset_id


def job_payload(job_config_path: Path, settings: NeurallsSettings) -> dict[str, Any]:
    """Return the identity-relevant content of a job, hashed after DLKit loads it.

    DLKit's loader inlines the referenced data profile and fills defaults, so
    comments, key order and profile file layout never matter, while any
    effective setting does.

    Args:
        job_config_path (Path): Job TOML.
        settings (NeurallsSettings): Runtime settings.

    Returns:
        dict[str, Any]: JSON-mode dump without reporting/location sections.
    """
    dumped = load_job_config(job_config_path, settings).model_dump(mode="json")
    return {k: v for k, v in dumped.items() if k not in _NON_IDENTITY_JOB_SECTIONS}


def training_identity(
    *,
    job_config_path: Path,
    data_config_path: Path,
    settings: NeurallsSettings,
) -> StageIdentity:
    """Derive the identity of one assignment's trained model.

    Args:
        job_config_path (Path): Job TOML.
        data_config_path (Path): Dataset TOML the job trains on.
        settings (NeurallsSettings): Runtime settings.

    Returns:
        StageIdentity: Changes iff the dataset content or the effective job
        settings change.

    Raises:
        FileNotFoundError: If the assignment's dataset has not been generated.
    """
    return StageIdentity.build(
        "training",
        {
            "dataset": current_dataset_digest(assignment_dataset_dir(data_config_path, settings)),
            "job": canonical_digest(job_payload(job_config_path, settings)),
        },
    )


def dataset_unchanged_since(
    tags: Mapping[str, str], data_config_path: Path, settings: NeurallsSettings
) -> bool:
    """Check that the dataset still matches the one a training run was keyed on.

    Guards the window between computing the key and finishing training: a
    dataset regenerated meanwhile means the run's tags no longer describe the
    data the model actually saw.

    Args:
        tags (Mapping[str, str]): Identity tags written on the run (see
            ``StageIdentity.tags``).
        data_config_path (Path): Dataset TOML the job trained on.
        settings (NeurallsSettings): Runtime settings.

    Returns:
        bool: ``True`` iff the dataset digest recorded in the run's components
        still equals the dataset's current digest.
    """
    recorded = json.loads(tags[IdentityTag.COMPONENTS])["dataset"]
    current = current_dataset_digest(assignment_dataset_dir(data_config_path, settings))
    return current.startswith(recorded)
