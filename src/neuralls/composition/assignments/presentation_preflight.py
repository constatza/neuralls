"""Builds the typed scientific presentation of a comparison before anything runs.

Every method is described from its resolved config; a POD-2G method additionally
carries the presentation of the dataset its basis was fitted on, read from that
dataset's manifest. Failures (a dataset without provenance, a neural method without a
display name, two indistinguishable methods) are raised here, before any MLflow run is
opened or any method executes.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    AMGPreconditionerConfig,
    ConcretePreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
)
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.shared.types import PreconditionerName

__all__ = [
    "compose_comparison_title",
    "describe_method",
    "prepare_comparison_presentation",
]


def describe_method(cfg: PreconditionerConfig) -> ConfiguredMethodPresentation:
    """Describe ``cfg``; a POD-2G method is qualified by its fit dataset, exactly once.

    The assignment and checkpoint identity of a fitted basis never reach the description.
    """
    method = describe_configured_preconditioner(cfg)
    match cfg:
        case AMGPreconditionerConfig(coarsening=PODCoarseningConfig() as pod):
            return DatasetQualifiedMethodPresentation(
                method=method, dataset=load_dataset_presentation(pod.dataset_dir)
            )
        case _:
            return StandaloneMethodPresentation(method=method)


def prepare_comparison_presentation(
    specs: Sequence[ConcretePreconditionerConfig], scope: ComparisonScopePresentation
) -> PreparedComparisonPresentation:
    """Describe every method of one comparison and validate the whole cohort."""
    candidates = tuple(
        MethodPresentationCandidate(PreconditionerName(spec.name), describe_method(spec))
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
