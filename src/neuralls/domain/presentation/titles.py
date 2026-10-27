"""Case and comparison titles: an editorial name, or the dataset a comparison ran on."""

from __future__ import annotations

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
