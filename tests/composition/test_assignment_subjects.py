"""Training and evaluation runs are about a typed subject, never an assignment id."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dlkit.infrastructure.config.job_config import FitJobConfig, TrainingJobConfig

from neuralls.composition.assignments import assignment_subjects
from neuralls.composition.assignments.assignment_subjects import (
    AssignmentInputs,
    IndistinguishableTrainingError,
    case_notation,
    require_distinct_training_labels,
    training_presentations,
    training_subject,
)
from neuralls.domain.presentation.methods import DatasetQualifiedMethodPresentation
from neuralls.domain.presentation.run_names import (
    EditorialTrainingSubject,
    TrainingPresentation,
    TrainingRunName,
    Untimed,
)
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    JobEntry,
    RegistryEntry,
)
from neuralls.platform.config.registry import ResolvedAssignmentBinding
from neuralls.platform.tracking.run_names import render_run_name
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.scientific_description.policy import NotationPolicy

PLAIN = NotationPolicy.default()
WORKSPACE = Path("/workspace")


def _binding(architecture: ValidatedLabel | None) -> ResolvedAssignmentBinding:
    return ResolvedAssignmentBinding(
        assignment_id="fit_job_1",
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


def _fit_job() -> MagicMock:
    job = MagicMock(spec=FitJobConfig)
    job.model = SimpleNamespace(rank=10)
    return job


def _subject(entry: AssignmentEntry) -> object:
    return training_subject(entry, cfg=MagicMock(), config_dir=Path("."), settings=MagicMock())


def _entry(display_name: str | None = None) -> AssignmentEntry:
    return AssignmentEntry(id="fit_job_1", dataset="ds_1", job="job_1", display_name=display_name)


def test_a_fit_job_is_the_pod_basis_trained_on_its_dataset(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(monkeypatch, pod_dataset_dir, _fit_job())

    subject = _subject(_entry())

    assert isinstance(subject, DatasetQualifiedMethodPresentation)
    text = render_run_name(TrainingRunName(subject, Untimed()), PLAIN)
    assert text == "POD-2G (r=10, ν₁=2, ν₂=2, ωₛ=auto) | Gaussian Solutions (μ=0, σ=1, N=8)"


def test_a_neural_job_is_its_architecture_trained_on_its_dataset(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(
        monkeypatch,
        pod_dataset_dir,
        MagicMock(spec=TrainingJobConfig),
        ValidatedLabel("Graph Net"),
    )

    subject = _subject(_entry())

    assert isinstance(subject, DatasetQualifiedMethodPresentation)
    text = render_run_name(TrainingRunName(subject, Untimed()), PLAIN)
    assert text == "Neural (arch=Graph Net) | Gaussian Solutions (μ=0, σ=1, N=8)"


def test_a_neural_job_without_an_architecture_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(monkeypatch, pod_dataset_dir, MagicMock(spec=TrainingJobConfig))

    with pytest.raises(ValueError, match="architecture"):
        _subject(_entry())


def test_an_explicit_display_name_is_an_editorial_title_beside_the_dataset(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(monkeypatch, pod_dataset_dir, _fit_job())

    subject = _subject(_entry("  Basis rank ten "))

    assert isinstance(subject, EditorialTrainingSubject)
    assert render_run_name(TrainingRunName(subject, Untimed()), PLAIN) == (
        "Basis rank ten | Atomic test | Gaussian Solutions (μ=0, σ=1, N=8)"
    )


def test_a_blank_display_name_is_no_editorial_title(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(monkeypatch, pod_dataset_dir, _fit_job())
    assert isinstance(_subject(_entry("   ")), DatasetQualifiedMethodPresentation)


def test_presentations_are_built_for_every_named_assignment_with_the_case_notation(
    monkeypatch: pytest.MonkeyPatch, pod_dataset_dir: Path
) -> None:
    _use_inputs(monkeypatch, pod_dataset_dir, _fit_job())
    cfg = CaseConfig(
        datasets=[RegistryEntry(id="ds_1", path=WORKSPACE / "data.toml")],
        jobs=[JobEntry(id="job_1", path=WORKSPACE / "job.toml")],
        assignments=[_entry()],
    )

    built = training_presentations(
        cfg, config_dir=WORKSPACE, settings=MagicMock(), assignment_ids=["fit_job_1"]
    )

    assert set(built) == {"fit_job_1"}
    assert built["fit_job_1"].notation == case_notation(cfg)


def test_assignments_that_render_alike_are_rejected_before_any_run_opens(
    training_presentation_of: Callable[[str], TrainingPresentation],
) -> None:
    same = training_presentation_of("POD-2G")

    with pytest.raises(IndistinguishableTrainingError, match="POD-2G"):
        require_distinct_training_labels(
            {"first": same, "second": training_presentation_of("POD-2G")}
        )


def test_assignments_with_different_models_are_distinct(
    training_presentation_of: Callable[[str], TrainingPresentation],
) -> None:
    require_distinct_training_labels(
        {"first": training_presentation_of("POD-2G"), "second": training_presentation_of("FNO")}
    )
