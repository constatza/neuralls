"""Case and comparison titles: an editorial name, or the dataset a comparison ran on."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from neuralls.domain.presentation.provenance import DatasetPresentation


@dataclass(frozen=True, slots=True)
class EditorialPresentation:
    """An explicit, validated human title for a case or job.

    It is cosmetic: it never replaces the scientific facts of a method or dataset.
    """

    title: str

    def __post_init__(self) -> None:
        """Reject blank, multi-line, control-character, or identifier-shaped titles."""
        if not self.title or self.title != self.title.strip():
            raise ValueError("EditorialPresentation.title must be non-empty and trimmed")
        if not self.title.isprintable():
            raise ValueError("EditorialPresentation.title must be a single printable line")
        if any("_" in word for word in self.title.split()):
            raise ValueError("EditorialPresentation.title must not contain identifier text")


@dataclass(frozen=True, slots=True)
class ComparisonScopePresentation:
    """What a comparison ran on."""

    dataset: DatasetPresentation


type ComparisonTitle = EditorialPresentation | ComparisonScopePresentation
"""A case/comparison title: an explicit editorial name, or the dataset it ran on."""


@dataclass(frozen=True, slots=True)
class CollectionTitle:
    """What a session ran on when its comparisons or assignments have different titles.

    Holds at least two distinct titles, so a one-title session never gets a collection.
    """

    parts: tuple[ComparisonTitle, ...]

    def __post_init__(self) -> None:
        """Require at least two distinct titles."""
        if len(set(self.parts)) != len(self.parts) or len(self.parts) < 2:
            raise ValueError("CollectionTitle requires at least two distinct titles")


type SessionTitle = ComparisonTitle | CollectionTitle
"""A batch session's title: its one shared title, or the collection of different ones."""


def session_title(titles: Sequence[ComparisonTitle]) -> SessionTitle:
    """Collapse the titles of a session's runs into the session's own title.

    Raises:
        ValueError: When the session has no title to compose from.
    """
    distinct = tuple(dict.fromkeys(titles))
    match distinct:
        case ():
            raise ValueError("a session needs at least one title")
        case (only,):
            return only
        case _:
            return CollectionTitle(distinct)
