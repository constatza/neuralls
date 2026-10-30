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


@dataclass(frozen=True, slots=True)
class SessionCounts:
    """How many systems a comparison session ran, and how they spread over RHS types and matrices."""

    n_systems: int
    n_rhs_types: int
    n_matrices: int

    def __post_init__(self) -> None:
        """Require a non-empty session whose systems cover at least one RHS type and matrix."""
        if min(self.n_systems, self.n_rhs_types, self.n_matrices) < 1:
            raise ValueError("SessionCounts requires at least one system, RHS type and matrix")


@dataclass(frozen=True, slots=True)
class SummarizedTitle:
    """A session over many systems: an optional shared title and how many systems it covered.

    The head is the one title every system shares; it is absent when the systems differ,
    so the session never enumerates its systems.
    """

    head: ComparisonTitle | None
    counts: SessionCounts


type SessionTitle = ComparisonTitle | CollectionTitle | SummarizedTitle
"""A batch session's title: its one shared title, a collection of different ones, or a summary."""


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


def summarized_session_title(
    titles: Sequence[ComparisonTitle], counts: SessionCounts
) -> SessionTitle:
    """The title of a comparison session: one lone system keeps its title, else a summary.

    Raises:
        ValueError: When the session has no title to compose from, or fewer titles than systems.
    """
    distinct = tuple(dict.fromkeys(titles))
    if not distinct:
        raise ValueError("a session needs at least one title")
    if len(titles) != counts.n_systems:
        raise ValueError("a session needs exactly one title per system")
    head = distinct[0] if len(distinct) == 1 else None
    if head is not None and counts.n_systems == 1:
        return head
    return SummarizedTitle(head=head, counts=counts)
