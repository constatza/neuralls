"""The figure index: one root document locating every figure of a report run tree.

The index is derived from the figure cards and holds only where each figure lives and what
it is; every parameter stays in the card. It is never edited by hand.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, assert_never

import tomli_w

from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_card import FIGURE_SUFFIX, FigureCard, card_name
from neuralls.platform.reporting.figure_metadata import (
    TABLE_SUFFIX,
    FigureKind,
    NoTable,
    SiblingTable,
)

FIGURE_INDEX_FILE = "figure_index.toml"

_REQUIRED_ENTRY_FIELDS: frozenset[str] = frozenset(
    {"run_id", "artifact_path", "card", "kind", "level", "scope"}
)


class InvalidFigureIndexError(ValueError):
    """A payload is not a complete figure index."""


class IncompleteFigureRecordsError(RuntimeError):
    """Uploaded figures, their cards and the index do not describe the same set of figures."""


@dataclass(frozen=True, slots=True)
class ArtifactLocation:
    """A file in a run's artifacts: the run and the path inside it."""

    run_id: str
    artifact_path: str

    @property
    def name(self) -> str:
        """The file name."""
        return PurePosixPath(self.artifact_path).name

    def sibling(self, name: str) -> ArtifactLocation:
        """The file ``name`` in the same run and directory."""
        return ArtifactLocation(
            self.run_id, PurePosixPath(self.artifact_path).with_name(name).as_posix()
        )


@dataclass(frozen=True, slots=True)
class FigureIndexEntry:
    """Where one figure and its card are, and what the figure is."""

    run_id: str
    artifact_path: str
    card: str
    kind: FigureKind
    level: ReportLevelName
    scope: str


def index_of(cards: Mapping[ArtifactLocation, FigureCard]) -> tuple[FigureIndexEntry, ...]:
    """One entry per card, in a deterministic order, locating the figure each card names."""
    entries = [
        FigureIndexEntry(
            run_id=location.run_id,
            artifact_path=location.sibling(card.figure).artifact_path,
            card=location.artifact_path,
            kind=card.metadata.kind,
            level=card.metadata.report_level,
            scope=card.scope_display,
        )
        for location, card in cards.items()
    ]
    return tuple(
        sorted(entries, key=lambda entry: (entry.level.value, entry.scope, entry.artifact_path))
    )


def _verify_table(
    figure: ArtifactLocation, card: FigureCard, uploaded: set[ArtifactLocation]
) -> None:
    match card.table:
        case SiblingTable(name=name):
            if figure.sibling(name) not in uploaded:
                raise IncompleteFigureRecordsError(f"figure {figure.artifact_path} has no table")
        case NoTable():
            pass
        case unreachable:
            assert_never(unreachable)


def verify_figure_records(
    uploaded: Sequence[ArtifactLocation],
    cards: Mapping[ArtifactLocation, FigureCard],
    index: Sequence[FigureIndexEntry],
) -> None:
    """Raise unless every uploaded figure has its card and the index lists exactly those figures.

    A figure is complete when a card with the same stem sits in the same run directory and
    names the figure's own file. A card without its figure, or an index entry without an
    uploaded figure, is as much a bug as a figure without a card.
    """
    figures = {location for location in uploaded if location.name.endswith(FIGURE_SUFFIX)}
    uploaded_set = set(uploaded)
    for figure in sorted(figures, key=lambda location: location.artifact_path):
        card_location = figure.sibling(card_name(figure.name))
        card = cards.get(card_location)
        if card is None or card_location not in uploaded_set:
            raise IncompleteFigureRecordsError(f"figure {figure.artifact_path} has no card")
        if card.figure != figure.name:
            raise IncompleteFigureRecordsError(
                f"card {card_location.artifact_path} describes {card.figure}, not {figure.name}"
            )
        _verify_table(figure, card, uploaded_set)
    tables = {location for location in uploaded if location.name.endswith(TABLE_SUFFIX)}
    expected_tables = {
        card_location.sibling(table.name)
        for card_location, card in cards.items()
        if isinstance(table := card.table, SiblingTable)
    }
    if tables != expected_tables:
        raise IncompleteFigureRecordsError("the uploaded tables are not exactly the group figures'")
    for card_location, card in cards.items():
        if card_location.sibling(card.figure) not in figures:
            raise IncompleteFigureRecordsError(
                f"card {card_location.artifact_path} has no uploaded figure {card.figure}"
            )
    indexed = [ArtifactLocation(entry.run_id, entry.artifact_path) for entry in index]
    if len(set(indexed)) != len(indexed) or set(indexed) != figures:
        raise IncompleteFigureRecordsError("the figure index does not list exactly the figures")
    for entry in index:
        expected = ArtifactLocation(entry.run_id, entry.artifact_path).sibling(
            card_name(PurePosixPath(entry.artifact_path).name)
        )
        if entry.card != expected.artifact_path:
            raise IncompleteFigureRecordsError(
                f"index entry {entry.artifact_path} names card {entry.card}, "
                f"not {expected.artifact_path}"
            )


def index_payload(entries: Sequence[FigureIndexEntry]) -> dict[str, Any]:
    """The index as TOML-ready data."""
    return {
        "figures": [
            {
                "run_id": entry.run_id,
                "artifact_path": entry.artifact_path,
                "card": entry.card,
                "kind": entry.kind.value,
                "level": entry.level.value,
                "scope": entry.scope,
            }
            for entry in entries
        ]
    }


def validate_figure_index(payload: Mapping[str, Any]) -> None:
    """Raise unless ``payload`` has a ``figures`` array of complete entries."""
    figures = payload.get("figures")
    if not isinstance(figures, list):
        raise InvalidFigureIndexError("figure index has no 'figures' array")
    for position, entry in enumerate(figures):
        if not isinstance(entry, Mapping):
            raise InvalidFigureIndexError(f"figure index entry {position} is not a table")
        missing = sorted(_REQUIRED_ENTRY_FIELDS - entry.keys())
        if missing:
            raise InvalidFigureIndexError(
                f"figure index entry {position} is missing fields: {', '.join(missing)}"
            )


def write_figure_index(path: Path, entries: Sequence[FigureIndexEntry]) -> None:
    """Write the index as TOML."""
    path.write_bytes(tomli_w.dumps(index_payload(entries)).encode("utf-8"))


def read_figure_index(path: Path) -> tuple[FigureIndexEntry, ...]:
    """Read and validate the index stored at ``path``."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_figure_index(payload)
    return tuple(
        FigureIndexEntry(
            run_id=str(entry["run_id"]),
            artifact_path=str(entry["artifact_path"]),
            card=str(entry["card"]),
            kind=FigureKind(entry["kind"]),
            level=ReportLevelName(entry["level"]),
            scope=str(entry["scope"]),
        )
        for entry in payload["figures"]
    )
