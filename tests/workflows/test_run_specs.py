"""Unit tests for typed MLflow run/tag builders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.composition.tracking.run_specs import (
    TrainingRunTags,
    build_benchmark_method_run_spec,
    build_evaluation_run,
    build_evaluation_tags,
    build_report_run_spec,
    build_session_run_spec,
    build_training_run_spec,
    comparison_run,
    system_qualifier,
)
from neuralls.domain.presentation.methods import DatasetQualifiedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    EditorialTrainingSubject,
    HeldOutMatrixPart,
    RhsDrawPart,
    RunTimestamp,
    SessionPhase,
    TrainingPresentation,
    Untimed,
)
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    EditorialPresentation,
    session_title,
)
from neuralls.domain.solver.models.provenance import HeldOutMatrix, SharedSingleMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import MatrixId
from tests.presentation_support import PreparedMethodFactory

PLAIN = NotationPolicy.default()
STAMP = "Thu 12 Mar 2026 - 12:00:00"
ISO = "2026-03-12T12:00:00"
DATASET_TEXT = "Test problem | CG Errors (k=20, N=100)"


@pytest.fixture
def trained(dataset_presentation: DatasetPresentation) -> TrainingPresentation:
    """A POD method trained on the test dataset."""
    return TrainingPresentation(
        DatasetQualifiedMethodPresentation(
            ScientificDescription("POD"),
            dataset_presentation,
            dataset_presentation.generation.generated_rows,
        ),
        PLAIN,
    )


def _entry(**kwargs: object) -> ComparisonRegistryEntry:
    return ComparisonRegistryEntry(
        id="cmp-1", matrix_dataset="solutions", rhs_source={"kind": "gaussian"}, **kwargs
    )


def _planned(
    entry: ComparisonRegistryEntry, *, matrix: int | None = None, ordinal: int = 0
) -> PlannedSystem:
    provenance = (
        SharedSingleMatrix(raw_matrix_index=0)
        if matrix is None
        else HeldOutMatrix(matrix_id=MatrixId(matrix), raw_matrix_index=0)
    )
    return PlannedSystem(
        entry=entry,
        matrix=MatrixRef(matrix_id=MatrixId(matrix or 0), row=0),
        rhs_row=None,
        rhs_ordinal=ordinal,
        provenance=provenance,
    )


def test_training_tags_keep_the_id_apart_from_the_rendered_label(
    trained: TrainingPresentation,
) -> None:
    tags = TrainingRunTags(
        phase="training",
        assignment_id="exp-1",
        dataset_id="dataset-1",
        job_id="job-1",
        subject=trained.subject,
        notation=PLAIN,
    )

    assert tags.as_mlflow_tags() == {
        "phase": "training",
        "assignment_id": "exp-1",
        "dataset_id": "dataset-1",
        "dataset": "Test problem | CG Errors (k=20, N=100)",
        "job_id": "job-1",
        "assignment_label": "POD | CG Errors (k=20, N=100)",
    }


def test_training_run_spec_is_named_by_what_it_trains(trained: TrainingPresentation) -> None:
    spec = build_training_run_spec(
        assignment_id="exp-1",
        dataset_id="dataset-1",
        job_id="job-1",
        presentation=trained,
        experiment_name="Train",
        paths=MlflowPaths("sqlite:///mlflow.db", None),
        workspace_root=Path("workspace"),
        timestamp=STAMP,
    )

    assert spec.experiment_name == "Train"
    assert run_name_text(spec.name) == "POD | CG Errors (k=20, N=100) | " + STAMP
    assert spec.tags["assignment_id"] == "exp-1"


def test_batched_training_run_name_has_no_timestamp(trained: TrainingPresentation) -> None:
    spec = build_training_run_spec(
        assignment_id="a",
        dataset_id="d",
        job_id="j",
        presentation=trained,
        experiment_name="Train",
        paths=MlflowPaths("sqlite:///mlflow.db", None),
        workspace_root=Path("workspace"),
        include_timestamp=False,
    )
    assert run_name_text(spec.name) == "POD | CG Errors (k=20, N=100)"


@pytest.mark.parametrize(
    ("timing", "suffix"),
    [(RunTimestamp(ISO), f" | {ISO}"), (Untimed(), "")],
    ids=["timed", "untimed"],
)
def test_comparison_run_name_identifies_matrix_set_and_ends_with_the_timestamp_when_timed(
    dataset_presentation: DatasetPresentation, timing: RunTimestamp | Untimed, suffix: str
) -> None:
    run = comparison_run(
        _planned(_entry()),
        ComparisonScopePresentation(dataset_presentation),
        dataset_presentation,
        PLAIN,
        timing,
    )

    assert run_name_text(run) == f"Comparison matrix set: {DATASET_TEXT}{suffix}"


def test_editorial_comparison_run_names_its_dataset_after_the_title(
    dataset_presentation: DatasetPresentation,
) -> None:
    run = comparison_run(
        _planned(_entry()),
        EditorialPresentation("Spectral"),
        dataset_presentation,
        PLAIN,
        Untimed(),
    )

    assert run_name_text(run) == f"Spectral | Comparison matrix set: {DATASET_TEXT}"


def test_sibling_systems_on_a_dataset_title_get_distinct_run_names(
    dataset_presentation: DatasetPresentation,
) -> None:
    title = ComparisonScopePresentation(dataset_presentation)
    names = []
    for matrix in (3, 4):
        entry = _entry(rhs_count=2)
        for ordinal in (0, 1):
            run = comparison_run(
                _planned(entry, matrix=matrix, ordinal=ordinal),
                title,
                dataset_presentation,
                PLAIN,
                Untimed(),
            )
            names.append(run_name_text(run))
    dataset = "Comparison matrix set: Test problem | CG Errors (k=20, N=100)"
    assert names == [
        f"{dataset} | matrix {matrix}, RHS {ordinal}" for matrix in (3, 4) for ordinal in (0, 1)
    ]


def test_a_single_draw_on_a_shared_matrix_needs_no_qualifier() -> None:
    entry = _entry()
    qualifier = system_qualifier(_planned(entry))
    assert qualifier.parts == ()


def test_a_held_out_matrix_with_several_draws_is_qualified_by_both() -> None:
    entry = _entry(rhs_count=3)
    qualifier = system_qualifier(_planned(entry, matrix=7, ordinal=2))
    assert qualifier.parts == (HeldOutMatrixPart(MatrixId(7)), RhsDrawPart(2))


def test_method_run_is_named_by_its_presentation_and_tagged_with_its_id(
    prepared_method: PreparedMethodFactory,
) -> None:
    run, tags = build_benchmark_method_run_spec(
        prepared=prepared_method("jacobi-1"),
        notation=PLAIN,
        config_digest="d" * 64,  # type: ignore[arg-type]
        system_ids=("s0",),
    )

    assert run_name_text(run) == "Jacobi"
    as_tags = tags.as_mlflow_tags()
    assert as_tags["method_id"] == "jacobi-1"
    assert as_tags["method_label"] == "Jacobi"


@pytest.mark.parametrize(
    ("phase", "stage", "title_text"),
    [
        ("session_training", "train", DATASET_TEXT),
        ("session_evaluation", "eval", DATASET_TEXT),
        ("session_comparison", "compare", f"Comparison matrix set: {DATASET_TEXT}"),
    ],
)
def test_session_run_spec_name_and_tags(
    phase: SessionPhase, stage: str, title_text: str, dataset_presentation: DatasetPresentation
) -> None:
    case_config_path = Path("workspace/cases/spectral/ffnn.toml")

    run, tags = build_session_run_spec(
        title=session_title([ComparisonScopePresentation(dataset_presentation)]),
        notation=PLAIN,
        case_config_path=case_config_path,
        experiment_name="spectral/ffnn",
        phase=phase,
        revision="d5d807fe0123456789abcdef0123456789abcdef-dirty",
        started=datetime(2026, 3, 12, 12, 0, 0),
    )

    assert run_name_text(run) == f"{stage} · 2026-03-12 12:00 · d5d807fe-dirty"
    assert tags.as_mlflow_tags() == {
        "phase": phase,
        "case_config": "ffnn",
        "case_config_path": case_config_path.as_posix(),
        "started_at": ISO,
        "neuralls.config": "spectral/ffnn",
        "neuralls.stage": stage,
        "title": title_text,
        "mlflow.note.content": title_text,
    }


@pytest.mark.parametrize(
    ("session_tags", "expected_tags"),
    [
        (
            {"neuralls.config": "spectral/old", "title": DATASET_TEXT},
            {
                "neuralls.config": "spectral/old",
                "title": DATASET_TEXT,
                "mlflow.note.content": DATASET_TEXT,
            },
        ),
        ({}, {"neuralls.config": "spectral/ffnn"}),
    ],
    ids=["session-tagged", "pre-tag-session"],
)
def test_report_run_spec_takes_config_and_title_from_its_session(
    session_tags: dict[str, str], expected_tags: dict[str, str]
) -> None:
    run, tags = build_report_run_spec(
        session_tags=session_tags,
        notation=PLAIN,
        config="spectral/ffnn",
        revision="d5d807fe0123456789abcdef0123456789abcdef",
        started=datetime(2026, 3, 12, 12, 0, 0),
    )

    assert run_name_text(run) == "report · 2026-03-12 12:00 · d5d807fe"
    assert tags == {"phase": "report", "neuralls.stage": "report", **expected_tags}


def test_evaluation_run_and_tags_use_the_same_subject(
    dataset_presentation: DatasetPresentation,
) -> None:
    presentation = TrainingPresentation(
        EditorialTrainingSubject(EditorialPresentation("Graph net run"), dataset_presentation),
        PLAIN,
    )

    run = build_evaluation_run(presentation)
    tags = build_evaluation_tags(
        assignment_id="exp-1",
        presentation=presentation,
        source_training_run_id="run-1",
        split_artifact="splits/split.json",
    ).as_mlflow_tags()

    assert run_name_text(run) == f"Graph net run | {DATASET_TEXT}"
    assert tags["assignment_label"] == f"Graph net run | {DATASET_TEXT}"
    assert tags["dataset"] == DATASET_TEXT
    assert tags["assignment_id"] == "exp-1"
