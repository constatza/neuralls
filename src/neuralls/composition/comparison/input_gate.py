"""Refuse to compare against datasets that generation has not produced and pinned.

Comparison consumes hashes: it selects systems out of an existing pinned test
fold and never generates, splits or trains anything itself. This gate proves,
before any planning, that each dataset it will read still matches what the case
config would generate today, so a stale or missing one fails with the stage to
run instead of silently comparing on the wrong data.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from neuralls.composition.comparison.config_assembler import resolve_dataset_dir
from neuralls.composition.generation.dataset_builder import is_dataset_reusable
from neuralls.composition.generation.default_services import FS_SPLIT_STORE
from neuralls.composition.generation.split_pinning import plan_split, split_is_current
from neuralls.composition.identity.generation import generation_identity
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.registry import resolve_dataset_config_path
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.dataset_readers import load_matrix_sample_index
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import REGENERATE_DATASETS_COMMAND
from neuralls.shared.errors import StageInputsStale


def require_current_inputs(
    case_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    dataset_ids: Iterable[str],
) -> None:
    """Check each dataset is generated under its current identity with a current split.

    Args:
        case_cfg: The validated case config.
        config_dir: Directory of the case config file.
        settings: Runtime settings.
        dataset_ids: Registry ids of the datasets comparison will read.

    Raises:
        StageInputsStale: Naming the dataset and the stage to run first.
    """
    for dataset_id in dict.fromkeys(dataset_ids):
        data_cfg = load_data_config(
            resolve_dataset_config_path(case_cfg, config_dir, dataset_id), settings
        )
        dataset_dir = resolve_dataset_dir(case_cfg, config_dir, dataset_id, settings)
        if not is_dataset_reusable(dataset_dir, generation_identity(data_cfg)):
            raise StageInputsStale(
                f"Dataset '{dataset_id}' at '{dataset_dir}' is missing or no longer matches "
                "its config and sources: run `neuralls generate` (or `neuralls run`) first."
            )
        if read_dataset_manifest(dataset_dir).provenance is None:
            raise StageInputsStale(
                f"Dataset '{dataset_id}' at '{dataset_dir}' has no scientific provenance (it "
                f"was generated before provenance was recorded): run "
                f"`{REGENERATE_DATASETS_COMMAND}` to rewrite it."
            )
        planned = plan_split(load_matrix_sample_index(dataset_dir), data_cfg.split)
        if not split_is_current(dataset_dir, planned, FS_SPLIT_STORE):
            raise StageInputsStale(
                f"Dataset '{dataset_id}' has no pinned split matching its [split] config: "
                "run `neuralls generate` (or `neuralls run`) first."
            )
