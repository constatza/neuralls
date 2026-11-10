"""Builds the typed scientific presentation of a comparison before anything runs.

Every method is described from its resolved config and validated data origin. A fit or
verified training dataset is read from its manifest. Failures (a dataset without provenance, a neural method without a
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
)
from neuralls.composition.preconditioners.descriptions import describe_configured_preconditioner
from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.platform.config.models.preconditioner import (
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
    cfg: PreconditionerConfig, origin: MethodDataOrigin, load_dataset: DatasetPresentationLoader
) -> ConfiguredMethodPresentation:
    """Describe a comparison method using only validated fit or training data."""
    method = describe_configured_preconditioner(cfg)
    match origin:
        case InlineFit(fit_dataset=directory):
            return DatasetQualifiedMethodPresentation(method, load_dataset(directory))
        case CheckpointBased(training_data=training_data):
            match training_data:
                case VerifiedTrainingDataset(dataset=directory):
                    return DatasetQualifiedMethodPresentation(method, load_dataset(directory))
                case TrainingDataUnavailable():
                    return StandaloneMethodPresentation(method=method)
                case unreachable:
                    assert_never(unreachable)
        case NoFitData():
            return StandaloneMethodPresentation(method=method)
        case unreachable:
            assert_never(unreachable)


def prepare_comparison_presentation(
    specs: Sequence[ConcretePreconditionerConfig],
    origins_by_method: Mapping[str, MethodDataOrigin],
    scope: ComparisonScopePresentation,
    load_dataset: DatasetPresentationLoader,
) -> PreparedComparisonPresentation:
    """Describe every method of one comparison and validate the whole cohort."""
    if set(origins_by_method) != {spec.name for spec in specs} or len(specs) != len(
        origins_by_method
    ):
        raise ValueError("method origins must match the resolved specs")
    candidates = tuple(
        MethodPresentationCandidate(
            PreconditionerName(spec.name),
            describe_method(spec, origins_by_method[spec.name], load_dataset),
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
