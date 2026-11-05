"""Offline cost resolution for preconditioner comparisons: dataset generation and training.

A comparison run never generates a dataset or trains a model itself; both happened as separate,
earlier pipeline stages. This module answers, for one preconditioner config, "what did building
its artifact cost, and do we still know that?" from records of those stages, never re-measured
here: the dataset's own manifest (`DatasetManifest.generation_duration_seconds`, stamped once at
generation time by composition/generation/dataset_builder.py) for generation, and the wall time
of the MLflow run each checkpoint was resolved from for training.

Training wall time spans the whole job lifecycle (data loading, epochs, checkpointing), not pure
fit time. A POD-2G basis fit inline from `dataset_dir` has no training stage here: that fit runs
inside the method's setup and is already measured there.

Known limitation: a POD-2G config's `dataset_dir` is a TOML-authoring
convention, not a verified invariant — nothing here checks that a
checkpoint-backed POD-2G's `dataset_dir` still matches the dataset its
checkpoint was actually fit from. Cross-checking that (e.g. a content-digest
check mirroring composition.generation.dataset_builder.is_dataset_reusable)
is real future work, intentionally out of scope here.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mlflow.tracking import MlflowClient

from neuralls.domain.solver.models.cost import (
    Known,
    NotApplicable,
    OfflineStages,
    StageCost,
    Unavailable,
)
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.checkpoint_refs import CheckpointRefBearing
from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import DatasetManifest
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.platform.tracking.run_duration import run_wall_seconds
from neuralls.shared.types import CostProvenance


def resolve_offline_cost(
    cfg: PreconditionerConfig, *, settings: NeurallsSettings, runs: MlflowClient
) -> OfflineStages:
    """Resolve both offline stages of `cfg`: the dataset generation and the checkpoint training.

    `runs` is the store holding the training runs the config's checkpoints were resolved from.
    """
    return OfflineStages(
        generation=resolve_generation_cost(cfg, settings=settings),
        training=resolve_training_cost(cfg, runs=runs),
    )


def resolve_training_cost(cfg: PreconditionerConfig, *, runs: MlflowClient) -> StageCost:
    """Resolve the wall time spent training the checkpoints `cfg` loads.

    `NotApplicable` for a config with no checkpoint (classical methods, and a POD-2G basis fit
    inline, whose fit is part of its measured setup). Otherwise the seconds of every distinct
    training run the checkpoints were resolved from, summed: two transfer operators trained in
    one run cost that run once. A checkpoint named by an explicit path has no run to read, so
    the cost is `Unavailable`, as it is when a run cannot be read or has not ended.
    """
    if not isinstance(cfg, CheckpointRefBearing):
        return NotApplicable()
    refs = cfg.checkpoint_refs()
    if not refs:
        return NotApplicable()
    n_unrecorded = 0
    run_ids: dict[str, None] = {}
    for _, ref in refs:
        match ref.resolved_run_id:
            case None:
                n_unrecorded += 1
            case run_id:
                run_ids[run_id] = None
    if n_unrecorded:
        return Unavailable(
            f"{n_unrecorded} of {len(refs)} checkpoints have no recorded training run"
        )
    spans = [run_wall_seconds(runs, run_id) for run_id in run_ids]
    reasons = [span.reason for span in spans if isinstance(span, Unavailable)]
    if reasons:
        return Unavailable("; ".join(reasons))
    return Known(
        sum(span.wall_time_seconds for span in spans if isinstance(span, Known)),
        None,
        CostProvenance.HISTORICAL,
    )


def resolve_generation_cost(cfg: PreconditionerConfig, *, settings: NeurallsSettings) -> StageCost:
    """Resolve the dataset-generation cost feeding `cfg`'s fit/train step.

    The one place that decides between the three outcomes. `NotApplicable` when
    generation doesn't apply to this preconditioner kind (classical/geometric AMG,
    standard/Jacobi/IC0 — no training data at all). Otherwise `Known` (HISTORICAL:
    a referenced dataset's manifest recorded generation_duration_seconds) or
    `Unavailable` (a dataset dir couldn't be resolved, or no manifest has the field).
    """
    dataset_dirs = _resolve_generation_dataset_dirs(cfg, settings=settings)
    if not dataset_dirs:
        return NotApplicable()
    manifests: list[DatasetManifest] = []
    for dataset_dir in dataset_dirs:
        try:
            manifests.append(read_dataset_manifest(dataset_dir))
        except FileNotFoundError, ValueError:
            continue
    return aggregate_generation_cost(manifests)


def aggregate_generation_cost(manifests: Iterable[DatasetManifest]) -> Known | Unavailable:
    """Sum the recorded generation times of `manifests`; `Unavailable` when none records one.

    Time adds up because every dataset was generated separately; peak memory takes the
    maximum because the generations never shared a process.
    """
    total_seconds = 0.0
    total_memory: int | None = None
    found_any = False
    for manifest in manifests:
        if manifest.generation_duration_seconds is None:
            continue
        total_seconds += manifest.generation_duration_seconds
        if manifest.generation_peak_memory_bytes is not None:
            total_memory = max(total_memory or 0, manifest.generation_peak_memory_bytes)
        found_any = True
    if not found_any:
        return Unavailable("no dataset manifest records a generation duration")
    return Known(total_seconds, total_memory, CostProvenance.HISTORICAL)


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
