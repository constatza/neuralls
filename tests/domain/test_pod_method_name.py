"""A dataset-qualified POD method is named by the snapshot source of its fit data."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neuralls.domain.presentation.methods import PreparedMethodPresentation, qualify_method
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.render import render_method
from neuralls.shared.scientific_description.catalog import RANK, STEPS, make_fact
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import LEGEND_PROFILE, NotationPolicy
from neuralls.shared.types import GenerationStrategyKind, PreconditionerName

FIT_ROWS = 90
"""Rows the fit read: fewer than the fixture dataset's 100 generated rows, as under a split.
A source-named POD states them as its ``N``; a dataset qualifier keeps the generated rows."""


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
        ((GenerationStrategyKind.CG_ERRORS,), "POD", "POD-CG"),
        ((GenerationStrategyKind.RESIDUALS,), "POD", "POD-CG"),
        ((GenerationStrategyKind.JACOBI_PROBES,), "POD", "POD-JP"),
        ((GenerationStrategyKind.GAUSSIAN_FORWARD,), "POD", "POD"),
        (
            (
                GenerationStrategyKind.CG_ERRORS,
                GenerationStrategyKind.JACOBI_PROBES,
            ),
            "POD",
            "POD",
        ),
        ((GenerationStrategyKind.CG_ERRORS,), "Neural", "Neural"),
    ],
)
def test_qualified_method_names_its_snapshot_source(
    dataset_presentation: DatasetPresentation,
    kinds: tuple[GenerationStrategyKind, ...],
    base_name: str,
    expected: str,
) -> None:
    dataset = _with_strategies(dataset_presentation, kinds)

    qualified = qualify_method(ScientificDescription(base_name), dataset, FIT_ROWS)

    assert qualified.method == ScientificDescription(expected)
    assert qualified.dataset == dataset


@pytest.mark.parametrize(
    ("kinds", "base_name", "expected"),
    [
        ((GenerationStrategyKind.CG_ERRORS,), "POD", "POD-CG (c=100, k=20, N=90)"),
        (
            (GenerationStrategyKind.CG_ERRORS, GenerationStrategyKind.JACOBI_PROBES),
            "POD",
            (
                "POD (c=100) | Mixture (N=100) + CG Errors (k=20, N=50)"
                " + Jacobi-Filtered Probes (k=20, N=50)"
            ),
        ),
        (
            (GenerationStrategyKind.GAUSSIAN_FORWARD,),
            "POD",
            "POD (c=100) | Gaussian Solutions (k=20, N=100)",
        ),
        ((GenerationStrategyKind.CG_ERRORS,), "Neural", "Neural (c=100) | CG Errors (k=20, N=100)"),
    ],
)
def test_only_a_source_named_pod_folds_its_strategy_facts_into_its_parenthesis(
    dataset_presentation: DatasetPresentation,
    kinds: tuple[GenerationStrategyKind, ...],
    base_name: str,
    expected: str,
) -> None:
    dataset = _with_strategies(dataset_presentation, kinds)
    method = ScientificDescription(base_name, facts=(make_fact(RANK, 100),))

    rendered = render_method(
        qualify_method(method, dataset, FIT_ROWS), LEGEND_PROFILE, NotationPolicy.default()
    )

    assert rendered.text == expected


def test_pod_jp_label_states_the_sweeps_as_k(dataset_presentation: DatasetPresentation) -> None:
    template = dataset_presentation.generation.strategies[0]
    probes = replace(
        template, kind=GenerationStrategyKind.JACOBI_PROBES, parameters=(make_fact(STEPS, 10),)
    )
    dataset = replace(
        dataset_presentation,
        generation=replace(dataset_presentation.generation, strategies=(probes,)),
    )
    method = ScientificDescription("POD", facts=(make_fact(RANK, 100),))

    rendered = render_method(
        qualify_method(method, dataset, FIT_ROWS), LEGEND_PROFILE, NotationPolicy.default()
    )

    assert rendered.text == "POD-JP (c=100, k=10, N=90)"


def _prepared_pod(
    dataset: DatasetPresentation, generated_rows: int, fit_rows: int
) -> PreparedMethodPresentation:
    cg = _with_strategies(dataset, (GenerationStrategyKind.CG_ERRORS,))
    strategy = replace(cg.generation.strategies[0], rows=generated_rows)
    generation = replace(cg.generation, strategies=(strategy,), generated_rows=generated_rows)
    method = ScientificDescription("POD", facts=(make_fact(RANK, 100),))
    configured = qualify_method(method, replace(cg, generation=generation), fit_rows)
    return PreparedMethodPresentation(
        PreconditionerName(f"pod-{generated_rows}-{fit_rows}"), configured, method.facts
    )


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        pytest.param((100, 90), (100, 80), False, id="one-dataset-two-split-ratios"),
        pytest.param((100, 90), (120, 90), True, id="two-datasets-one-fit-size"),
    ],
)
def test_pod_compact_identity_agrees_with_its_rendered_label(
    dataset_presentation: DatasetPresentation,
    first: tuple[int, int],
    second: tuple[int, int],
    same: bool,
) -> None:
    a, b = (_prepared_pod(dataset_presentation, *rows) for rows in (first, second))
    policy = NotationPolicy.default()

    labels = (render_method(m.compact, LEGEND_PROFILE, policy).text for m in (a, b))

    assert (a.compact_identity() == b.compact_identity(), len(set(labels)) == 1) == (same, same)
