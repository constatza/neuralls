"""Builds the typed scientific presentation of a comparison before anything runs.

Every method is described from its resolved config and validated data origin. A fit or
verified training dataset is read from its manifest, and the rows the fit read from its
pinned split. Failures (a dataset without provenance, a neural method without a
display name, two indistinguishable methods) are raised here, before any MLflow run is
opened or any method executes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import assert_never

from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    MethodDataOrigin,
    NoFitData,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
    fold_rows,
)
from neuralls.composition.generation.ports import SplitStore
from neuralls.composition.generation.split_pinning import pinned_train_rows
from neuralls.composition.preconditioners.descriptions import describe_configured_preconditioner
from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
    qualify_method,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.shared.types import PreconditionerName

__all__ = [
    "DatasetPresentationLoader",
    "compose_comparison_title",
    "describe_method",
    "prepare_comparison_presentation",
]


type DatasetPresentationLoader = Callable[[Path], DatasetPresentation]
"""Reads the presentation of the dataset at a directory; composition wires the manifest reader."""


def describe_method(
    cfg: PreconditionerConfig,
    origin: MethodDataOrigin,
    load_dataset: DatasetPresentationLoader,
    splits: SplitStore,
) -> ConfiguredMethodPresentation:
    """Describe a comparison method using only validated fit or training data.

    A fit is qualified by the rows it read: an inline POD fit the train fold of the split
    pinned now (cut to its ``n_snapshots``), a verified checkpoint the folds its job read of
    the split its run recorded. Each row is one snapshot vector, so for a POD basis these
    are the columns of its snapshot matrix.

    Raises:
        StageInputsStale: An inline fit's dataset has no pinned split.
    """
    method = describe_configured_preconditioner(cfg)
    match origin:
        case InlineFit(fit_dataset=directory):
            dataset = load_dataset(directory)
            train_rows = len(pinned_train_rows(directory, splits))
            return qualify_method(method, dataset, _inline_fit_rows(cfg, train_rows))
        case CheckpointBased(training_data=training_data):
            match training_data:
                case VerifiedTrainingDataset(dataset=directory, split=pinned, folds=folds):
                    return qualify_method(
                        method, load_dataset(directory), fold_rows(pinned.split, folds)
                    )
                case TrainingDataUnavailable():
                    return StandaloneMethodPresentation(method=method)
                case unreachable:
                    assert_never(unreachable)
        case NoFitData():
            return StandaloneMethodPresentation(method=method)
        case unreachable:
            assert_never(unreachable)


def _inline_fit_rows(cfg: PreconditionerConfig, train_rows: int) -> int:
    """The train rows an inline fit reads; only a POD coarsening fits inline."""
    match cfg:
        case AMGPreconditionerConfig(coarsening=PODCoarseningConfig() as pod):
            return pod.inline_fit_rows(train_rows)
        case _:
            raise ValueError("only a POD coarsening is fit inline")


def prepare_comparison_presentation(
    specs: Sequence[ConcretePreconditionerConfig],
    origins_by_method: Mapping[str, MethodDataOrigin],
    scope: ComparisonScopePresentation,
    load_dataset: DatasetPresentationLoader,
    splits: SplitStore,
) -> PreparedComparisonPresentation:
    """Describe every method of one comparison and validate the whole cohort."""
    if set(origins_by_method) != {spec.name for spec in specs} or len(specs) != len(
        origins_by_method
    ):
        raise ValueError("method origins must match the resolved specs")
    candidates = tuple(
        MethodPresentationCandidate(
            PreconditionerName(spec.name),
            describe_method(spec, origins_by_method[spec.name], load_dataset, splits),
        )
        for spec in specs
    )
    return prepare_presentations(UnvalidatedComparisonPresentation(candidates, scope))


def compose_comparison_title(
    editorial_name: str | None, dataset: DatasetPresentation
) -> ComparisonTitle:
    """An explicit editorial name when configured, otherwise the dataset the comparison ran on.

    Never falls back to a registry id; an invalid editorial name raises.
    """
    if editorial_name is None:
        return ComparisonScopePresentation(dataset)
    return EditorialPresentation(editorial_name.strip())
