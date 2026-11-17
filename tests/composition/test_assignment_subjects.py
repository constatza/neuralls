"""Training and evaluation runs are about a typed subject, never an assignment id."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dlkit.infrastructure.config.job_config import TrainingJobConfig

from neuralls.composition.assignments import assignment_subjects
from neuralls.composition.assignments.assignment_subjects import (
    AssignmentInputs,
    IndistinguishableTrainingError,
    case_notation,
    pod_spec_from_assignment,
    require_distinct_training_labels,
    training_presentations,
    training_subject,
)
from neuralls.composition.preconditioners.descriptions import describe_configured_preconditioner
from neuralls.domain.presentation.methods import DatasetQualifiedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    EditorialTrainingSubject,
    TrainingPresentation,
    TrainingRunName,
    Untimed,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    JobEntry,
    RegistryEntry,
)
from neuralls.platform.config.models.pod_job import PODJobConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.config.registry import ResolvedAssignmentBinding
from neuralls.platform.tracking.run_names import render_run_name, render_training_label
from neuralls.shared.scientific_description.catalog import INNER_PRODUCT, make_fact
from neuralls.shared.scientific_description.models import ScientificFact, ValidatedLabel
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import InnerProductKind
from tests.presentation_support import datasets_by_dir, train_splits

PLAIN = NotationPolicy.default()
FIT_DIR = Path("pod-fit")
WORKSPACE = Path("workspace")


def _binding(architecture: ValidatedLabel | None) -> ResolvedAssignmentBinding:
    return ResolvedAssignmentBinding(
        assignment_id="assignment_1",
        dataset_id="ds_1",
        job_id="job_1",
        data_config_path=WORKSPACE / "data.toml",
        job_config_path=WORKSPACE / "job.toml",
        architecture=architecture,
    )


def _use_inputs(
    monkeypatch: pytest.MonkeyPatch,
    dataset_dir: Path,
    job: object,
    architecture: ValidatedLabel | None = None,
) -> None:
    inputs = AssignmentInputs(
        binding=_binding(architecture),
        job=job,  # type: ignore[arg-type]
        dataset_dir=dataset_dir,
    )
    monkeypatch.setattr(assignment_subjects, "resolve_assignment_inputs", lambda *a, **k: inputs)


def _pod_recipe(**pod: object) -> PODJobConfig:
    return PODJobConfig.model_validate({"run": {"type": "pod"}, "pod": {"rank": 10, **pod}})


def _neural_job() -> object:
    return MagicMock(spec=TrainingJobConfig)


GRAPH_NET = ValidatedLabel("Graph Net")


def _subject(entry: AssignmentEntry, fit_presentation: DatasetPresentation) -> object:
    return training_subject(
        entry,
        cfg=MagicMock(),
        config_dir=Path("."),
        settings=MagicMock(),
        load_dataset=datasets_by_dir({FIT_DIR: fit_presentation}),
        splits=train_splits(FIT_DIR),
    )


def _entry(display_name: str | None = None) -> AssignmentEntry:
    return AssignmentEntry(
        id="assignment_1", dataset="ds_1", job="job_1", display_name=display_name
    )


def test_a_pod_recipe_has_no_training_subject(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _pod_recipe())

    with pytest.raises(TypeError):
        _subject(_entry(), pod_fit_presentation)


def test_a_neural_job_is_its_architecture_trained_on_its_dataset(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _neural_job(), GRAPH_NET)

    subject = _subject(_entry(), pod_fit_presentation)

    assert isinstance(subject, DatasetQualifiedMethodPresentation)
    text = render_run_name(TrainingRunName(subject, Untimed()), PLAIN)
    assert text == "Neural (arch=Graph Net) | Gaussian Solutions (μ=0, σ=1, N=8)"


def test_a_neural_job_without_an_architecture_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _neural_job())

    with pytest.raises(ValueError, match="architecture"):
        _subject(_entry(), pod_fit_presentation)


def test_an_explicit_display_name_is_an_editorial_title_beside_the_dataset(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _neural_job(), GRAPH_NET)

    subject = _subject(_entry("  Basis rank ten "), pod_fit_presentation)

    assert isinstance(subject, EditorialTrainingSubject)
    assert render_run_name(TrainingRunName(subject, Untimed()), PLAIN) == (
        "Basis rank ten | Atomic test | Gaussian Solutions (μ=0, σ=1, N=8)"
    )


def test_a_blank_display_name_is_no_editorial_title(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _neural_job(), GRAPH_NET)
    assert isinstance(
        _subject(_entry("   "), pod_fit_presentation), DatasetQualifiedMethodPresentation
    )


def test_presentations_are_built_for_every_named_assignment_with_the_case_notation(
    monkeypatch: pytest.MonkeyPatch, pod_fit_presentation: DatasetPresentation
) -> None:
    _use_inputs(monkeypatch, FIT_DIR, _neural_job(), GRAPH_NET)
    cfg = CaseConfig(
        datasets=[RegistryEntry(id="ds_1", path=WORKSPACE / "data.toml")],
        jobs=[JobEntry(id="job_1", path=WORKSPACE / "job.toml")],
        assignments=[_entry()],
    )

    built = training_presentations(
        cfg,
        config_dir=WORKSPACE,
        settings=MagicMock(),
        assignment_ids=["assignment_1"],
        load_dataset=datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        splits=train_splits(FIT_DIR),
    )

    assert set(built) == {"assignment_1"}
    assert built["assignment_1"].notation == case_notation(cfg)


def test_assignments_that_render_alike_are_rejected_before_any_run_opens(
    training_presentation_of: Callable[[str], TrainingPresentation],
) -> None:
    same = training_presentation_of("POD")

    with pytest.raises(IndistinguishableTrainingError) as error:
        require_distinct_training_labels({"first": same, "second": training_presentation_of("POD")})
    assert (error.value.text, error.value.assignment_ids) == (
        render_training_label(same),
        ("first", "second"),
    )


def test_assignments_with_different_models_are_distinct(
    training_presentation_of: Callable[[str], TrainingPresentation],
) -> None:
    require_distinct_training_labels(
        {"first": training_presentation_of("POD"), "second": training_presentation_of("FNO")}
    )


@pytest.mark.parametrize(
    ("pod", "fit"),
    [
        ({}, {}),
        (
            {"inner_product": "energy", "n_snapshots": 4},
            {"inner_product": InnerProductKind.ENERGY, "n_snapshots": 4},
        ),
    ],
    ids=["defaults", "energy-cut"],
)
def test_a_pod_recipe_is_a_pod_method_fitted_on_its_dataset(
    pod: dict[str, object], fit: dict[str, object]
) -> None:
    spec = pod_spec_from_assignment(_pod_recipe(**pod), dataset_dir=FIT_DIR)

    assert spec == AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=10, **fit),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("inner_product", "facts"),
    [("euclidean", ()), ("energy", (make_fact(INNER_PRODUCT, "energy"),))],
)
def test_only_an_energy_pod_method_states_its_inner_product(
    inner_product: str, facts: tuple[ScientificFact, ...]
) -> None:
    spec = pod_spec_from_assignment(_pod_recipe(inner_product=inner_product), dataset_dir=FIT_DIR)

    described = describe_configured_preconditioner(spec).facts
    assert tuple(fact for fact in described if fact.key == INNER_PRODUCT) == facts
