"""Value-level checks of the per-method history legend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
from tests.provenance_support import provenance_for

from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetPresentation, DatasetSourceProvenance
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.platform.reporting.method_figures import _history_legend
from neuralls.shared.rhs_recipe import GaussianRecipe, RawRhsRecipe, RawSourceIdentity
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, SystemId


@pytest.fixture
def notation() -> NotationPolicy:
    return NotationPolicy.default()


@pytest.fixture
def history_inputs(
    dataset_presentation: DatasetPresentation,
) -> tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]]:
    first = DatasetId("first")
    second = DatasetId("second")
    other = replace(
        dataset_presentation,
        source=DatasetSourceProvenance(ScientificDescription("Other problem")),
    )
    gaussian = GaussianRecipe(0.0, 1.0)
    shifted = GaussianRecipe(1.0, 1.0)
    systems = provenance_for(
        (
            SystemTag(first, gaussian, MatrixId(0), 0, SystemId("a")),
            SystemTag(first, gaussian, MatrixId(1), 0, SystemId("b")),
            SystemTag(first, shifted, MatrixId(0), 1, SystemId("c")),
            SystemTag(second, gaussian, MatrixId(0), 0, SystemId("d")),
        )
    )
    return systems, {first: dataset_presentation, second: other}


@pytest.fixture
def raw_source_inputs(
    dataset_presentation: DatasetPresentation,
) -> tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]]:
    dataset = DatasetId("first")
    systems = provenance_for(
        (
            SystemTag(
                dataset, RawRhsRecipe(RawSourceIdentity(0, "a")), MatrixId(0), 0, SystemId("a")
            ),
            SystemTag(
                dataset, RawRhsRecipe(RawSourceIdentity(0, "b")), MatrixId(0), 1, SystemId("b")
            ),
        )
    )
    return systems, {dataset: dataset_presentation}


@pytest.fixture
def incomplete_datasets(
    history_inputs: tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]],
) -> Mapping[DatasetId, DatasetPresentation]:
    _, datasets = history_inputs
    return {DatasetId("first"): datasets[DatasetId("first")]}


@pytest.mark.parametrize(
    ("plotted", "expected_groups", "expected_text"),
    [
        (("a", "b"), {"a": "a", "b": "a"}, ("Gaussian",)),
        (("a", "c"), {"a": "a", "c": "c"}, ("Gaussian", "Gaussian (μ=1, σ=1)")),
        (("a", "d"), {"a": "a", "d": "d"}, ("Test problem", "Other problem")),
    ],
)
def test_history_legend_names_only_varying_group_context(
    history_inputs: tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]],
    notation: NotationPolicy,
    plotted: tuple[str, ...],
    expected_groups: dict[str, str],
    expected_text: tuple[str, ...],
) -> None:
    systems, datasets = history_inputs

    result = _history_legend(systems, plotted, notation, datasets)

    assert result.group_for_system == expected_groups
    texts = tuple(entry.text for entry in result.legend.entries.values())
    assert len(texts) == len(expected_text)
    for text, expected in zip(texts, expected_text, strict=True):
        assert expected in text
        assert "Comparison matrix set:" not in text
        assert "matrix " not in text
        assert "RHS " not in text


def test_history_legend_combines_dataset_and_recipe_only_when_both_vary(
    history_inputs: tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]],
    notation: NotationPolicy,
) -> None:
    systems, datasets = history_inputs

    result = _history_legend(systems, ("a", "c", "d"), notation, datasets)

    assert result.group_for_system == {"a": "a", "c": "c", "d": "d"}
    texts = tuple(entry.text for entry in result.legend.entries.values())
    assert "Test problem" in texts[0] and "Gaussian" in texts[0]
    assert "Test problem" in texts[1] and "Gaussian" in texts[1]
    assert "Other problem" in texts[2] and "Gaussian" in texts[2]


def test_history_legend_rejects_missing_dataset(
    history_inputs: tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]],
    incomplete_datasets: Mapping[DatasetId, DatasetPresentation],
    notation: NotationPolicy,
) -> None:
    systems, _ = history_inputs

    with pytest.raises(MissingMethodPresentationError):
        _history_legend(systems, ("d",), notation, incomplete_datasets)


def test_distinct_raw_sources_never_share_a_visible_label(
    raw_source_inputs: tuple[tuple[SystemProvenance, ...], Mapping[DatasetId, DatasetPresentation]],
    notation: NotationPolicy,
) -> None:
    systems, datasets = raw_source_inputs

    result = _history_legend(systems, ("a", "b"), notation, datasets)

    assert tuple(entry.text for entry in result.legend.entries.values()) == (
        "Stored RHS vector (sample 0), variant 1",
        "Stored RHS vector (sample 0), variant 2",
    )
