"""Dataset-generation cost resolution for preconditioner comparisons.

A comparison run never generates a dataset itself — that already happened as
a separate, earlier pipeline stage (composition/generation). This module
answers, for one preconditioner config, "how much did generating its training
data cost, and do we still know that?" — read back from the dataset's own
manifest (DatasetManifest.generation_duration_seconds, stamped once at
generation time by composition/generation/dataset_builder.py), never
re-measured here.

Known limitation: a POD-2G config's `dataset_dir` is a TOML-authoring
convention, not a verified invariant — nothing here checks that a
checkpoint-backed POD-2G's `dataset_dir` still matches the dataset its
checkpoint was actually fit from. Cross-checking that (e.g. a content-digest
check mirroring composition.generation.dataset_builder.is_dataset_reusable)
is real future work, intentionally out of scope here.
"""

from __future__ import annotations

from pathlib import Path

from neuralls.domain.solver.models.result import StageCost
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    CheckpointRefBearing,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.types import CostProvenance


def resolve_generation_cost(
    cfg: PreconditionerConfig, *, settings: NeurallsSettings
) -> StageCost | None:
    """Resolve the dataset-generation cost feeding `cfg`'s fit/train step.

    Returns None when generation doesn't apply to this preconditioner kind
    (classical/geometric AMG, standard/Jacobi/IC0 — no training data at all).
    Otherwise always HISTORICAL (every referenced dataset's manifest has a
    recorded generation_duration_seconds) or UNAVAILABLE (a dataset dir
    couldn't be resolved, or its manifest lacks the field).
    """
    dataset_dirs = _resolve_generation_dataset_dirs(cfg, settings=settings)
    if not dataset_dirs:
        return None
    total_seconds = 0.0
    total_memory: int | None = None
    found_any = False
    for dataset_dir in dataset_dirs:
        try:
            manifest = read_dataset_manifest(dataset_dir)
        except FileNotFoundError, ValueError:
            continue
        if manifest.generation_duration_seconds is not None:
            total_seconds += manifest.generation_duration_seconds
            if manifest.generation_peak_memory_bytes is not None:
                total_memory = max(total_memory or 0, manifest.generation_peak_memory_bytes)
            found_any = True
    if not found_any:
        return StageCost(0.0, None, CostProvenance.UNAVAILABLE)
    return StageCost(total_seconds, total_memory, CostProvenance.HISTORICAL)


def _resolve_generation_dataset_dirs(
    cfg: PreconditionerConfig, *, settings: NeurallsSettings
) -> set[Path]:
    """Every dataset directory feeding `cfg`'s fit/train step, deduplicated."""
    if isinstance(cfg, AMGPreconditionerConfig) and isinstance(
        cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig
    ):
        return {Path(cfg.coarsening.dataset_dir)}
    if isinstance(cfg, CheckpointRefBearing):
        dirs: set[Path] = set()
        for _, ref in cfg.checkpoint_refs():
            if ref.data_config_path is None:
                continue
            data_cfg = load_data_config(ref.data_config_path, settings)
            resolve_dataset_identity(data_cfg=data_cfg, config_path=ref.data_config_path)
            data_dir = data_cfg.output.data_dir
            if data_dir is None:
                data_dir = settings.processed_dir
            dirs.add(Path(data_dir) / data_cfg.id)
        return dirs
    return set()
