"""Historical generation and training costs for a validated method data origin."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import assert_never

from dlkit.infrastructure.types.split import IndexSplit
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    MethodDataOrigin,
    NoFitData,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
)
from neuralls.composition.generation.ports import SplitStore
from neuralls.domain.solver.models.cost import (
    FitFolds,
    GenerationCost,
    Known,
    NotApplicable,
    OfflineStages,
    StageCost,
    TrainShare,
    TrainShareOfGeneration,
    Unavailable,
)
from neuralls.platform.storage.manifest import DatasetManifest
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.platform.tracking.run_duration import run_wall_seconds
from neuralls.shared.types import CostProvenance


def resolve_offline_cost(
    origin: MethodDataOrigin,
    *,
    runs: MlflowClient,
    splits: SplitStore,
    run_costs: Mapping[str, Known | Unavailable] | None = None,
) -> OfflineStages:
    """Attribute offline stages to the prepared method's validated dependencies."""
    return OfflineStages(
        generation=resolve_generation_cost(origin, splits),
        training=resolve_training_cost(origin, runs=runs, run_costs=run_costs),
    )


def resolve_training_cost(
    origin: MethodDataOrigin,
    *,
    runs: MlflowClient,
    run_costs: Mapping[str, Known | Unavailable] | None = None,
) -> StageCost:
    """Sum each distinct resolved training run once; an unknown run stays unknown."""
    if isinstance(origin, NoFitData | InlineFit):
        return NotApplicable()
    refs = origin.checkpoint_refs
    n_unrecorded = sum(ref.resolved_run_id is None for ref in refs)
    if n_unrecorded:
        return Unavailable(
            f"{n_unrecorded} of {len(refs)} checkpoints have no recorded training run"
        )
    run_ids = dict.fromkeys(ref.resolved_run_id for ref in refs if ref.resolved_run_id is not None)
    spans = [
        run_costs[run_id] if run_costs is not None else run_wall_seconds(runs, run_id)
        for run_id in run_ids
    ]
    reasons = [span.reason for span in spans if isinstance(span, Unavailable)]
    if reasons:
        return Unavailable("; ".join(reasons))
    return Known(
        sum(span.wall_time_seconds for span in spans if isinstance(span, Known)),
        None,
        CostProvenance.HISTORICAL,
    )


def resolve_generation_cost(origin: MethodDataOrigin, splits: SplitStore) -> GenerationCost:
    """Charge the share of its dataset's generation that the method's fit read.

    Generation produced every row, but a fit reads only some folds of the dataset's pinned
    split: an inline POD fit the train fold of the split pinned now, a checkpoint the folds its
    job read of the split its run recorded. The held-out test rows are generated for
    evaluation, so they are never charged to the artifact.
    """
    match origin:
        case NoFitData():
            return NotApplicable()
        case InlineFit(fit_dataset=dataset_dir):
            return _share_of_generation(dataset_dir, splits.read(dataset_dir), FitFolds.TRAIN)
        case CheckpointBased(training_data=training_data):
            match training_data:
                case VerifiedTrainingDataset(dataset=dataset_dir, split=pinned, folds=folds):
                    return _share_of_generation(dataset_dir, pinned.split, folds)
                case TrainingDataUnavailable(reason=reason):
                    return Unavailable(reason)
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _share_of_generation(
    dataset_dir: Path, split: IndexSplit | None, folds: FitFolds
) -> GenerationCost:
    whole = _dataset_generation_cost(dataset_dir)
    if not isinstance(whole, Known):
        return whole
    if split is None:
        return Unavailable("the fit dataset has no pinned split to take the train share of")
    if not split.train:
        return Unavailable("the fit dataset's pinned split has no train rows")
    total_rows = len(split.train) + len(split.validation) + len(split.test)
    return TrainShareOfGeneration(
        whole, TrainShare.of_rows(_read_rows(split, folds), total_rows, folds)
    )


def _read_rows(split: IndexSplit, folds: FitFolds) -> int:
    match folds:
        case FitFolds.TRAIN:
            return len(split.train)
        case FitFolds.TRAIN_AND_VALIDATION:
            return len(split.train) + len(split.validation)
        case unreachable:
            assert_never(unreachable)


def _dataset_generation_cost(dataset_dir: Path) -> Known | Unavailable:
    try:
        manifest = read_dataset_manifest(dataset_dir)
    except FileNotFoundError, ValueError:
        return Unavailable("no dataset manifest records a generation duration")
    return aggregate_generation_cost((manifest,))


def aggregate_generation_cost(manifests: Iterable[DatasetManifest]) -> Known | Unavailable:
    """Sum recorded generation times; unknown durations never count as zero."""
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
