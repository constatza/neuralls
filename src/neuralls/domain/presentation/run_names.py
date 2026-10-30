"""What an MLflow run is named after, as typed values; text is rendered at the tracking sink.

A run name carries the presentation objects that identify the run (a title, a method, a
dataset) plus the small typed pieces that keep sibling runs apart (a timestamp, the matrix
and right-hand-side draw of one system). It never carries an identifier or a rendered
string, so no caller can name a run after a registry id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    EditorialPresentation,
    SessionTitle,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import MatrixId


@dataclass(frozen=True, slots=True)
class RunTimestamp:
    """The wall-clock stamp that tells repeated runs of the same case apart."""

    text: str

    def __post_init__(self) -> None:
        """Reject a blank, untrimmed, or multi-line stamp."""
        if not self.text or self.text != self.text.strip() or not self.text.isprintable():
            raise ValueError("RunTimestamp must be one non-blank printable line")


@dataclass(frozen=True, slots=True)
class Untimed:
    """The run sits under a session run that already carries the timestamp."""


type RunStamp = RunTimestamp | Untimed


@dataclass(frozen=True, slots=True)
class HeldOutMatrixPart:
    """The system is on one held-out matrix of a multi-matrix dataset."""

    matrix_id: MatrixId


@dataclass(frozen=True, slots=True)
class RhsDrawPart:
    """The system is one of several right-hand-side draws on its matrix."""

    ordinal: int

    def __post_init__(self) -> None:
        """Require a non-negative draw position."""
        if self.ordinal < 0:
            raise ValueError("RhsDrawPart.ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class SystemQualifier:
    """What tells one system's run apart from its sibling systems; empty for a sole system."""

    parts: tuple[HeldOutMatrixPart | RhsDrawPart, ...]


@dataclass(frozen=True, slots=True)
class EditorialTrainingSubject:
    """A trained model named by an explicit editorial title, on the dataset it trained on."""

    editorial: EditorialPresentation
    dataset: DatasetPresentation


type TrainingSubject = DatasetQualifiedMethodPresentation | EditorialTrainingSubject
"""What a training or evaluation run is about: a dataset-trained method, or an editorial name."""


def training_dataset(subject: TrainingSubject) -> DatasetPresentation:
    """The dataset a training subject was trained on."""
    match subject:
        case DatasetQualifiedMethodPresentation(dataset=dataset):
            return dataset
        case EditorialTrainingSubject(dataset=dataset):
            return dataset
        case _:
            assert_never(subject)


@dataclass(frozen=True, slots=True)
class TrainingPresentation:
    """What a training or evaluation run is about, and the notation it is rendered with."""

    subject: TrainingSubject
    notation: NotationPolicy


@dataclass(frozen=True, slots=True)
class SessionRunName:
    """The batch parent run: the case title and when the batch started."""

    title: SessionTitle
    started: RunTimestamp


@dataclass(frozen=True, slots=True)
class EditorialComparison:
    """A comparison named by an explicit editorial title, on the dataset it ran on."""

    editorial: EditorialPresentation
    dataset: DatasetPresentation


type ComparisonSubject = ComparisonScopePresentation | EditorialComparison
"""What a comparison run is about: the dataset alone, or an editorial name beside its dataset."""


@dataclass(frozen=True, slots=True)
class ComparisonRunName:
    """One comparison system's run: its subject, its sibling qualifier, and an optional stamp."""

    subject: ComparisonSubject
    system: SystemQualifier
    stamp: RunStamp


@dataclass(frozen=True, slots=True)
class MethodRunName:
    """One method's run: the method's own configured presentation and nothing else."""

    method: ConfiguredMethodPresentation


@dataclass(frozen=True, slots=True)
class TrainingRunName:
    """One training run: what was trained and, for a standalone run, when."""

    subject: TrainingSubject
    stamp: RunStamp


@dataclass(frozen=True, slots=True)
class EvaluationRunName:
    """One evaluation run: the evaluated model and the dataset it is evaluated on."""

    subject: TrainingSubject


type RunName = (
    SessionRunName | ComparisonRunName | MethodRunName | TrainingRunName | EvaluationRunName
)


@dataclass(frozen=True, slots=True)
class NamedRun:
    """A run name together with the case's notation; the tracking sink renders it."""

    name: RunName
    notation: NotationPolicy
