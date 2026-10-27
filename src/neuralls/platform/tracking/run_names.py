"""Render a typed run name to the plain Unicode text MLflow shows.

Composition builds run names from presentation objects; this adapter is the only place
they become text, so a run name cannot be assembled from an id or an unvalidated string.
"""

from __future__ import annotations

from dataclasses import replace
from typing import assert_never

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.render import render_dataset, render_method, render_title
from neuralls.domain.presentation.run_names import (
    ComparisonRunName,
    ComparisonSubject,
    EditorialComparison,
    EditorialTrainingSubject,
    EvaluationRunName,
    HeldOutMatrixPart,
    MethodRunName,
    NamedRun,
    RunName,
    RunStamp,
    RunTimestamp,
    SessionRunName,
    SystemQualifier,
    TrainingPresentation,
    TrainingRunName,
    TrainingSubject,
    Untimed,
)
from neuralls.domain.presentation.titles import (
    CollectionTitle,
    ComparisonScopePresentation,
    SessionTitle,
)
from neuralls.shared.scientific_description.policy import (
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)

RUN_NAME_PROFILE = RenderProfile(name="run_name", include_context=False)
"""Run names list the compact facts on one line; MLflow does not wrap them."""

_PART_SEPARATOR = " | "
_COLLECTION_SEPARATOR = "; "


def _stamped(text: str, stamp: RunStamp) -> str:
    match stamp:
        case RunTimestamp(text=stamped):
            return f"{text}{_PART_SEPARATOR}{stamped}"
        case Untimed():
            return text
        case _:
            assert_never(stamp)


def _title_text(title: SessionTitle, profile: RenderProfile, policy: NotationPolicy) -> str:
    match title:
        case CollectionTitle(parts=parts):
            return _COLLECTION_SEPARATOR.join(
                sorted(_title_text(part, profile, policy) for part in parts)
            )
        case _:
            return render_title(title, profile, policy).text


def _qualifier_text(qualifier: SystemQualifier) -> str:
    return ", ".join(
        f"matrix {part.matrix_id}" if isinstance(part, HeldOutMatrixPart) else f"RHS {part.ordinal}"
        for part in qualifier.parts
    )


def _comparison_subject_text(
    subject: ComparisonSubject, profile: RenderProfile, policy: NotationPolicy
) -> str:
    """The subject's title: the dataset alone, or an editorial name and its dataset."""
    match subject:
        case ComparisonScopePresentation():
            return render_title(subject, profile, policy).text
        case EditorialComparison(editorial=editorial, dataset=dataset):
            return (
                f"{editorial.title}{_PART_SEPARATOR}{render_dataset(dataset, profile, policy).text}"
            )
        case _:
            assert_never(subject)


def _subject_text(subject: TrainingSubject, profile: RenderProfile, policy: NotationPolicy) -> str:
    """The subject's title: a dataset-trained method, or an editorial name and its dataset."""
    match subject:
        case DatasetQualifiedMethodPresentation():
            return render_method(subject, profile, policy).text
        case EditorialTrainingSubject(editorial=editorial, dataset=dataset):
            return (
                f"{editorial.title}{_PART_SEPARATOR}{render_dataset(dataset, profile, policy).text}"
            )
        case _:
            assert_never(subject)


def render_session_title(title: SessionTitle, notation: NotationPolicy) -> str:
    """Render a title alone, for a ``*_label`` tag."""
    return _title_text(title, RUN_NAME_PROFILE, replace(notation, target=RenderTarget.PLAIN))


def render_training_subject(subject: TrainingSubject, notation: NotationPolicy) -> str:
    """Render what a training or evaluation run is about, for a ``*_label`` tag."""
    return _subject_text(subject, RUN_NAME_PROFILE, replace(notation, target=RenderTarget.PLAIN))


def render_dataset_label(dataset: DatasetPresentation, notation: NotationPolicy) -> str:
    """Render a dataset as plain text, for logs and summaries."""
    return render_dataset(
        dataset, RUN_NAME_PROFILE, replace(notation, target=RenderTarget.PLAIN)
    ).text


def render_method_label(method: ConfiguredMethodPresentation, notation: NotationPolicy) -> str:
    """Render a configured method as plain text, for logs and summaries."""
    return render_method(
        method, RUN_NAME_PROFILE, replace(notation, target=RenderTarget.PLAIN)
    ).text


def render_training_label(presentation: TrainingPresentation) -> str:
    """Render what a training presentation is about, for logs, summaries and report legends."""
    return render_training_subject(presentation.subject, presentation.notation)


def render_run_name(name: RunName, notation: NotationPolicy) -> str:
    """Render ``name`` as plain text under the case's symbol overrides."""
    policy = replace(notation, target=RenderTarget.PLAIN)
    profile = RUN_NAME_PROFILE
    match name:
        case SessionRunName(title=title, started=started):
            return _stamped(_title_text(title, profile, policy), started)
        case ComparisonRunName(subject=subject, system=system, stamp=stamp):
            text = _comparison_subject_text(subject, profile, policy)
            if system.parts:
                text = f"{text}{_PART_SEPARATOR}{_qualifier_text(system)}"
            return _stamped(text, stamp)
        case MethodRunName(method=method):
            return render_method(method, profile, policy).text
        case TrainingRunName(subject=subject, stamp=stamp):
            return _stamped(_subject_text(subject, profile, policy), stamp)
        case EvaluationRunName(subject=subject):
            return _subject_text(subject, profile, policy)
        case _:
            assert_never(name)


def run_name_text(run: NamedRun) -> str:
    """The text MLflow (or dlkit, for the runs it opens) shows for ``run``."""
    return render_run_name(run.name, run.notation)
