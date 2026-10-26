"""Scientific presentation is built and validated before any run opens or method executes."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralls.composition.assignments.comparison_batch import (
    ResolvedComparisonSpecs,
    _ComparisonResolutionContext,
    _neural_spec_from_assignment,
    _PreparedComparisonExecution,
    run_comparison_batch,
)
from neuralls.composition.assignments.presentation_preflight import (
    compose_comparison_title,
    describe_method,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    PlannedSystem,
)
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.models.experiments import AssignmentEntry
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    PODCoarseningConfig,
)
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description import (
    LEGEND_PROFILE,
    ComparisonScopePresentation,
    DatasetPresentation,
    DatasetQualifiedMethodPresentation,
    DuplicateMethodPresentationError,
    EditorialPresentation,
    NotationPolicy,
    PreparedComparisonPresentation,
    ScientificDescription,
    StandaloneMethodPresentation,
    render_method,
    render_title,
)
from neuralls.shared.scientific_description.presentation import DatasetSourceProvenance
from tests.workflows.comparison_support import MLFLOW_MODULE
from tests.workflows.test_comparison_workflow import _mock_cfg, _write_experiments_config

PLAIN = NotationPolicy.default()


@pytest.fixture
def pod_dataset_dir(tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec) -> Path:
    """A real generated dataset whose manifest carries scientific provenance."""
    target = tmp_path / "pod-fit"
    build_dataset(
        atomic_source,
        atomic_spec,
        str(target),
        source_provenance=DatasetSourceProvenance(ScientificDescription("Atomic test")),
    )
    return target


@pytest.fixture
def stale_dataset_dir(tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec) -> Path:
    """A generated dataset written without provenance."""
    target = tmp_path / "stale"
    build_dataset(atomic_source, atomic_spec, str(target))
    return target


def _pod(dataset_dir: Path, rank: int, assignment: str = "fit-job") -> AMGPreconditionerConfig:
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=rank, assignment=assignment)
    )


def test_pod_method_is_qualified_by_its_fit_dataset_exactly_once(pod_dataset_dir: Path) -> None:
    presentation = describe_method(_pod(pod_dataset_dir, 10))

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
        [_pod(pod_dataset_dir, 10), _pod(pod_dataset_dir, 100)],
        ComparisonScopePresentation(dataset_presentation),
    )
    labels = [render_method(m.compact, LEGEND_PROFILE, PLAIN).text for m in prepared.methods]
    assert labels[0] != labels[1]
    assert "r=10," in labels[0] and "r=100," in labels[1]


def test_dataset_without_provenance_is_rejected(stale_dataset_dir: Path) -> None:
    with pytest.raises(MissingDatasetProvenanceError, match="Regenerate"):
        describe_method(_pod(stale_dataset_dir, 10))


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
        build_dataset(atomic_source, atomic_spec, str(target), source_provenance=problem)
        copies.append(_pod(target, 10, assignment=f"fit-{name}"))

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


def test_auto_neural_stub_carries_only_the_explicit_job_display_name() -> None:
    entry = AssignmentEntry(id="a1", dataset="d", job="j")
    named = _neural_spec_from_assignment(entry, extra_input_names=(), architecture_name="Graph Net")
    unnamed = _neural_spec_from_assignment(entry, extra_input_names=(), architecture_name=None)
    assert named.display_name == "Graph Net"
    assert unnamed.display_name is None


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_a_presentation_failure_aborts_the_batch_before_any_run_opens(
    tmp_path: Path, prepared_presentation: PreparedComparisonPresentation
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)
    seen: list[str] = []

    def _prepare(
        cfg: object,
        planned: PlannedSystem,
        context: _ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> ComparisonOutcome | _PreparedComparisonExecution:
        seen.append(planned.entry.id)
        if len(seen) == 2:
            raise MissingDatasetProvenanceError(tmp_path / "stale")
        return _PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(specs=[], warnings=(), checkpoint_digests={}),
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=context.settings,
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_prepare,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=_mock_cfg(),
        ),
        patch("neuralls.composition.assignments.comparison_batch._execute_all") as execute,
        patch(MLFLOW_MODULE) as mock_mlflow,
        pytest.raises(MissingDatasetProvenanceError),
    ):
        run_comparison_batch(experiments_config, ComparisonParams())

    mock_mlflow.start_run.assert_not_called()
    execute.assert_not_called()


def test_prepared_batch_item_count_must_match_the_plan() -> None:
    from neuralls.composition.assignments.comparison_batch import PreparedComparisonBatch

    with pytest.raises(ValueError, match="one item per planned system"):
        PreparedComparisonBatch(items=(), planned=(object(),))  # type: ignore[arg-type]
