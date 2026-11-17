"""Scientific presentation is built and validated before any run opens or method executes."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.composition.assignments.assignment_subjects import neural_spec_from_assignment
from neuralls.composition.assignments.presentation_preflight import (
    compose_comparison_title,
    describe_method,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    InlineFit,
    NoFitData,
    PinnedSplit,
    TrainingDataUnavailable,
    VerifiedTrainingDataset,
)
from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    DuplicateMethodPresentationError,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.render import render_method, render_title
from neuralls.domain.presentation.run_names import MethodRunName, NamedRun
from neuralls.domain.presentation.titles import ComparisonScopePresentation, EditorialPresentation
from neuralls.domain.solver.models.cost import FitFolds
from neuralls.domain.solver.models.figure_scope import DatasetScope
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.experiments import AssignmentEntry
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend, scope_label
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.scientific_description.policy import LEGEND_PROFILE, NotationPolicy
from tests.fakes.split_store import InMemorySplitStore
from tests.presentation_support import (
    NO_DATASET_READS,
    datasets_by_dir,
    fixture_origins,
    pod_spec,
    train_splits,
)

PLAIN = NotationPolicy.default()
FIT_DIR = Path("pod-fit")


def test_pod_method_is_qualified_by_its_fit_dataset_exactly_once(
    pod_fit_presentation: DatasetPresentation,
) -> None:
    presentation = describe_method(
        pod_spec(FIT_DIR, 10),
        InlineFit(FIT_DIR),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        train_splits(FIT_DIR),
    )

    assert isinstance(presentation, DatasetQualifiedMethodPresentation)
    text = render_method(presentation, LEGEND_PROFILE, PLAIN).text
    assert text == "POD (c=10) | Gaussian Solutions (μ=0, σ=1, N=8)"


FIT_SPLIT = IndexSplit(train=(0, 1, 2, 3, 4), validation=(5, 6), test=(7,), predict=None)
"""Five train, two validation and one test row of the eight-row fit dataset."""


def _verified(folds: FitFolds) -> CheckpointBased:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, checkpoint_path=Path("basis.pt"))
    )
    split = PinnedSplit(FIT_SPLIT, "sha256:split")
    return CheckpointBased(
        tuple(ref for _, ref in cfg.checkpoint_refs()),
        VerifiedTrainingDataset(FIT_DIR, "sha256:abcdef0123456789", split, folds),
    )


@pytest.mark.parametrize(
    ("snapshots", "origin", "fit_rows"),
    [
        (-1, InlineFit(FIT_DIR), 5),
        (3, InlineFit(FIT_DIR), 3),
        (-1, _verified(FitFolds.TRAIN), 5),
        (-1, _verified(FitFolds.TRAIN_AND_VALIDATION), 7),
    ],
    ids=["inline-train-fold", "inline-cut-to-n-snapshots", "fit-job-train", "train-and-validation"],
)
def test_a_fit_is_qualified_by_the_rows_of_the_folds_it_read(
    pod_fit_presentation: DatasetPresentation,
    snapshots: int,
    origin: InlineFit | CheckpointBased,
    fit_rows: int,
) -> None:
    """One dataset row is one snapshot vector, so the fit folds' rows are the columns of V."""
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=2, n_snapshots=snapshots)
    )

    presentation = describe_method(
        cfg,
        origin,
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        InMemorySplitStore({FIT_DIR: FIT_SPLIT}),
    )

    assert isinstance(presentation, DatasetQualifiedMethodPresentation)
    assert presentation.fit_rows == fit_rows


def test_ordinary_methods_are_standalone() -> None:
    cfg = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    assert isinstance(
        describe_method(cfg, NoFitData(), NO_DATASET_READS, InMemorySplitStore()),
        StandaloneMethodPresentation,
    )


def test_unverified_checkpoint_pod_has_no_fit_dataset_qualifier() -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, checkpoint_path=Path("basis.pt"))
    )
    assert isinstance(
        describe_method(
            cfg,
            CheckpointBased(
                tuple(ref for _, ref in cfg.checkpoint_refs()),
                TrainingDataUnavailable("no training run"),
            ),
            NO_DATASET_READS,
            InMemorySplitStore(),
        ),
        StandaloneMethodPresentation,
    )


@pytest.fixture
def neural_checkpoint() -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        architecture="Net",
        resolved_checkpoint_path=Path("network.pt"),
        resolved_run_id="run-1",
    )


@pytest.mark.parametrize(
    ("verified", "expected_label"),
    [
        (True, "Neural (arch=Net) | Gaussian Solutions (μ=0, σ=1, N=8)"),
        (False, "Neural (arch=Net)"),
    ],
    ids=["verified-training-dataset", "unavailable-training-dataset"],
)
def test_neural_method_label_follows_checkpoint_provenance(
    neural_checkpoint: NeuralPreconditionerConfig,
    pod_fit_presentation: DatasetPresentation,
    dataset_presentation: DatasetPresentation,
    verified: bool,
    expected_label: str,
) -> None:
    ref = tuple(ref for _, ref in neural_checkpoint.checkpoint_refs())
    training = (
        VerifiedTrainingDataset(
            FIT_DIR,
            "sha256:abcdef0123456789",
            PinnedSplit(
                IndexSplit(train=(0,), validation=(), test=(1,), predict=None), "sha256:split"
            ),
            FitFolds.TRAIN_AND_VALIDATION,
        )
        if verified
        else TrainingDataUnavailable("training run has no dataset identity")
    )
    prepared = prepare_comparison_presentation(
        [neural_checkpoint],
        {neural_checkpoint.name: CheckpointBased(ref, training)},
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}) if verified else NO_DATASET_READS,
        InMemorySplitStore(),
    )
    method = prepared.methods[0].compact
    label = method_legend({neural_checkpoint.name: method}, PLAIN).label(neural_checkpoint.name)
    assert run_name_text(NamedRun(MethodRunName(method), PLAIN)) == label
    assert label == expected_label
    assert (
        scope_label(DatasetScope(dataset_presentation), PLAIN)
        == "Comparison matrix set: Test problem | CG Errors (k=20, N=100)"
    )


def test_pod_ranks_over_one_dataset_are_distinguished_by_rank(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    specs = [pod_spec(FIT_DIR, 10), pod_spec(FIT_DIR, 100)]
    prepared = prepare_comparison_presentation(
        specs,
        fixture_origins(specs),
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        train_splits(FIT_DIR),
    )
    labels = [render_method(m.compact, LEGEND_PROFILE, PLAIN).text for m in prepared.methods]
    assert labels == [
        "POD (c=10) | Gaussian Solutions (μ=0, σ=1, N=8)",
        "POD (c=100) | Gaussian Solutions (μ=0, σ=1, N=8)",
    ]


def test_a_dataset_that_cannot_be_described_is_rejected() -> None:
    def unprovenanced(directory: Path) -> DatasetPresentation:
        raise MissingDatasetProvenanceError(directory)

    with pytest.raises(MissingDatasetProvenanceError) as error:
        describe_method(
            pod_spec(FIT_DIR, 10), InlineFit(FIT_DIR), unprovenanced, train_splits(FIT_DIR)
        )
    assert error.value.dataset_dir == FIT_DIR


def test_indistinguishable_methods_are_rejected_before_execution(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    """Two fit assignments over identical datasets have distinct ids but one presentation."""
    copy_a, copy_b = Path("copy-a"), Path("copy-b")
    copies = [
        pod_spec(copy_a, 10, assignment="fit-copy-a"),
        pod_spec(copy_b, 10, assignment="fit-copy-b"),
    ]

    with pytest.raises(DuplicateMethodPresentationError) as caught:
        prepare_comparison_presentation(
            copies,
            fixture_origins(copies),
            ComparisonScopePresentation(dataset_presentation),
            datasets_by_dir({copy_a: pod_fit_presentation, copy_b: pod_fit_presentation}),
            train_splits(copy_a, copy_b),
        )

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
        "Test problem | CG Errors (k=20, N=100)"
    )


@pytest.mark.parametrize("name", ["", "   ", "bad_name", "two\nlines"])
def test_invalid_editorial_titles_are_rejected(
    name: str, dataset_presentation: DatasetPresentation
) -> None:
    with pytest.raises(ValueError):
        compose_comparison_title(name, dataset_presentation)


def test_auto_neural_stub_carries_only_the_explicit_job_architecture() -> None:
    entry = AssignmentEntry(id="a1", dataset="d", job="j", display_name="Search (ReLU, bias)")
    named = neural_spec_from_assignment(
        entry, extra_input_names=(), architecture=ValidatedLabel("Graph Net")
    )
    unnamed = neural_spec_from_assignment(entry, extra_input_names=(), architecture=None)
    assert named.architecture == "Graph Net"
    assert unnamed.architecture is None


def test_prepared_batch_item_count_must_match_the_plan() -> None:
    from neuralls.composition.assignments.comparison_prepared import (
        PreparedComparisonBatch,
    )

    with pytest.raises(ValueError, match="one item per planned system"):
        PreparedComparisonBatch(items=(), planned=(object(),), split_records=())  # type: ignore[arg-type]
