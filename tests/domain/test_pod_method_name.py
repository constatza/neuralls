"""A dataset-qualified POD method is named by the snapshot source of its fit data."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neuralls.domain.presentation.methods import qualify_method
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.types import GenerationStrategyKind


def _with_strategies(
    dataset: DatasetPresentation, kinds: tuple[GenerationStrategyKind, ...]
) -> DatasetPresentation:
    template = dataset.generation.strategies[0]
    rows = dataset.generation.generated_rows // len(kinds)
    strategies = tuple(replace(template, kind=kind, rows=rows) for kind in kinds)
    generation = replace(
        dataset.generation, strategies=strategies, generated_rows=rows * len(kinds)
    )
    return replace(dataset, generation=generation)


@pytest.mark.parametrize(
    ("kinds", "base_name", "expected"),
    [
        ((GenerationStrategyKind.GAUSSIAN_RESIDUALS,), "POD", "POD-CG"),
        ((GenerationStrategyKind.RESIDUALS,), "POD", "POD-CG"),
        ((GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,), "POD", "POD-JP"),
        ((GenerationStrategyKind.GAUSSIAN_FORWARD,), "POD", "POD"),
        (
            (
                GenerationStrategyKind.GAUSSIAN_RESIDUALS,
                GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,
            ),
            "POD",
            "POD",
        ),
        ((GenerationStrategyKind.GAUSSIAN_RESIDUALS,), "Neural", "Neural"),
    ],
)
def test_qualified_method_names_its_snapshot_source(
    dataset_presentation: DatasetPresentation,
    kinds: tuple[GenerationStrategyKind, ...],
    base_name: str,
    expected: str,
) -> None:
    dataset = _with_strategies(dataset_presentation, kinds)

    qualified = qualify_method(ScientificDescription(base_name), dataset)

    assert qualified.method == ScientificDescription(expected)
    assert qualified.dataset == dataset
