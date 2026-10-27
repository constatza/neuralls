"""Scientific presentation is built and validated before any run opens or method executes."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.assignments.comparison_batch import (
    _neural_spec_from_assignment,
)
from neuralls.composition.assignments.presentation_preflight import (
    compose_comparison_title,
    describe_method,
    prepare_comparison_presentation,
)
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.config.models.experiments import AssignmentEntry
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
)
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError
from neuralls.shared.scientific_description import (
    LEGEND_PROFILE,
    ComparisonScopePresentation,
    DatasetPresentation,
    DatasetQualifiedMethodPresentation,
    DuplicateMethodPresentationError,
    EditorialPresentation,
    NotationPolicy,
    ScientificDescription,
    StandaloneMethodPresentation,
    ValidatedLabel,
    render_method,
    render_title,
)
from neuralls.shared.scientific_description.presentation import DatasetSourceProvenance
from tests.dataset_support import build_test_dataset
from tests.presentation_support import pod_spec

PLAIN = NotationPolicy.default()


def test_pod_method_is_qualified_by_its_fit_dataset_exactly_once(pod_dataset_dir: Path) -> None:
    presentation = describe_method(pod_spec(pod_dataset_dir, 10))

    assert isinstance(presentation, DatasetQualifiedMethodPresentation)
    text = render_method(presentation, LEGEND_PROFILE, PLAIN).text
    assert text.startswith("POD-2G (r=10, ")
    assert text.count("|") == 1
    assert text.count("Gaussian Solutions") == 1
    assert "fit-job" not in text
    assert "pod-fit" not in text


def test_ordinary_methods_are_standalone() -> None:
    cfg = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    assert isinstance(describe_method(cfg), StandaloneMethodPresentation)


def test_pod_ranks_over_one_dataset_are_distinguished_by_rank(
    pod_dataset_dir: Path, dataset_presentation: DatasetPresentation
) -> None:
    prepared = prepare_comparison_presentation(
        [pod_spec(pod_dataset_dir, 10), pod_spec(pod_dataset_dir, 100)],
        ComparisonScopePresentation(dataset_presentation),
    )
    labels = [render_method(m.compact, LEGEND_PROFILE, PLAIN).text for m in prepared.methods]
    assert labels[0] != labels[1]
    assert "r=10," in labels[0] and "r=100," in labels[1]


def test_dataset_without_provenance_is_rejected(stale_dataset_dir: Path) -> None:
    with pytest.raises(MissingDatasetProvenanceError, match="Regenerate"):
        describe_method(pod_spec(stale_dataset_dir, 10))


def test_indistinguishable_methods_are_rejected_before_execution(
    tmp_path: Path,
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dataset_presentation: DatasetPresentation,
) -> None:
    """Two fit assignments over identical datasets have distinct ids but one presentation."""
    problem = DatasetSourceProvenance(ScientificDescription("Atomic test"))
    copies = []
    for name in ("copy-a", "copy-b"):
        target = tmp_path / name
        build_test_dataset(atomic_source, atomic_spec, str(target), source_provenance=problem)
        copies.append(pod_spec(target, 10, assignment=f"fit-{name}"))

    with pytest.raises(DuplicateMethodPresentationError) as caught:
        prepare_comparison_presentation(copies, ComparisonScopePresentation(dataset_presentation))

    assert len(caught.value.method_ids) == 2


def test_title_is_the_editorial_name_when_configured(
    dataset_presentation: DatasetPresentation,
) -> None:
    title = compose_comparison_title("  Spectral comparison ", dataset_presentation)
    assert title == EditorialPresentation("Spectral comparison")
    assert render_title(title, LEGEND_PROFILE, PLAIN).text == "Spectral comparison"


def test_title_falls_back_to_the_dataset_never_to_an_id(
    dataset_presentation: DatasetPresentation,
) -> None:
    title = compose_comparison_title(None, dataset_presentation)
    assert isinstance(title, ComparisonScopePresentation)
    assert render_title(title, LEGEND_PROFILE, PLAIN).text == (
        "Test problem | Gaussian CG Residuals (k=20, N=100)"
    )


@pytest.mark.parametrize("name", ["", "   ", "bad_name", "two\nlines"])
def test_invalid_editorial_titles_are_rejected(
    name: str, dataset_presentation: DatasetPresentation
) -> None:
    with pytest.raises(ValueError):
        compose_comparison_title(name, dataset_presentation)


def test_auto_neural_stub_carries_only_the_explicit_job_architecture() -> None:
    entry = AssignmentEntry(id="a1", dataset="d", job="j", display_name="Search (ReLU, bias)")
    named = _neural_spec_from_assignment(
        entry, extra_input_names=(), architecture=ValidatedLabel("Graph Net")
    )
    unnamed = _neural_spec_from_assignment(entry, extra_input_names=(), architecture=None)
    assert named.architecture == "Graph Net"
    assert unnamed.architecture is None


def test_prepared_batch_item_count_must_match_the_plan() -> None:
    from neuralls.composition.assignments.comparison_batch import PreparedComparisonBatch

    with pytest.raises(ValueError, match="one item per planned system"):
        PreparedComparisonBatch(items=(), planned=(object(),))  # type: ignore[arg-type]
