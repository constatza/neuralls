"""The figure index: one root document locating every figure of a report run tree.

The index is derived from the figure cards and holds only where each figure lives and what
it is; every parameter stays in the card. It is never edited by hand.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, assert_never

import tomli_w

from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_card import FigureCard, card_name
from neuralls.platform.reporting.figure_formats import split_figure_file
from neuralls.platform.reporting.figure_metadata import (
    TABLE_SUFFIX,
    FigureKind,
    NoTable,
    SiblingTable,
)
from neuralls.shared.errors import ReasonedError

FIGURE_INDEX_FILE = "figure_index.toml"

_REQUIRED_ENTRY_FIELDS: frozenset[str] = frozenset(
    {"run_id", "artifact_paths", "card", "kind", "level", "scope"}
)


class InvalidFigureIndexReason(StrEnum):
    """How a payload fails to be a complete figure index."""

    NO_SPLIT_CITATION = "no_split_citation"
    NO_FIGURES_ARRAY = "no_figures_array"
    ENTRY_NOT_A_TABLE = "entry_not_a_table"
    ENTRY_MISSING_FIELDS = "entry_missing_fields"


class InvalidFigureIndexError(ReasonedError[InvalidFigureIndexReason], ValueError):
    """A payload is not a complete figure index."""


class FigureIndexEntryFieldsMissingError(InvalidFigureIndexError):
    """A figure index entry lacks required fields.

    Attributes:
        missing: The names that are absent, in the order the message lists them.
    """

    def __init__(self, detail: str, missing: tuple[str, ...]) -> None:
        super().__init__(InvalidFigureIndexReason.ENTRY_MISSING_FIELDS, detail)
        self.missing = missing

    def _constructor_args(self) -> tuple[object, ...]:
        return (self.detail, self.missing)


class IncompleteFigureRecordsReason(StrEnum):
    """Which pair of uploaded figures, cards, tables and index entries disagrees."""

    MISSING_TABLE = "missing_table"
    MISSING_CARD = "missing_card"
    CARD_DESCRIBES_OTHER_FIGURE = "card_describes_other_figure"
    MISSING_FILES = "missing_files"
    TABLES_MISMATCH = "tables_mismatch"
    CARD_WITHOUT_FIGURE = "card_without_figure"
    INDEX_FIGURES_MISMATCH = "index_figures_mismatch"
    INDEX_ENTRY_CARD_MISMATCH = "index_entry_card_mismatch"


class IncompleteFigureRecordsError(ReasonedError[IncompleteFigureRecordsReason], RuntimeError):
    """Uploaded figures, their cards and the index do not describe the same set of figures."""


class FigureFilesMissingError(IncompleteFigureRecordsError):
    """A figure card lists sibling files that were not uploaded.

    Attributes:
        missing: The names that are absent, in the order the message lists them.
    """

    def __init__(self, detail: str, missing: tuple[str, ...]) -> None:
        super().__init__(IncompleteFigureRecordsReason.MISSING_FILES, detail)
        self.missing = missing

    def _constructor_args(self) -> tuple[object, ...]:
        return (self.detail, self.missing)


class FigureCardMissingError(IncompleteFigureRecordsError):
    """An uploaded figure has no uploaded card beside it.

    Attributes:
        figure: The figure without a card.
    """

    def __init__(self, figure: ArtifactLocation) -> None:
        super().__init__(
            IncompleteFigureRecordsReason.MISSING_CARD, f"figure {figure.artifact_path} has no card"
        )
        self.figure = figure

    def _constructor_args(self) -> tuple[object, ...]:
        return (self.figure,)


class CardDescribesOtherFigureError(IncompleteFigureRecordsError):
    """The card beside a figure describes a figure of another name.

    Attributes:
        card: The card at fault.
        described: The figure stem the card names.
        figure: The figure name the card sits beside.
    """

    def __init__(self, card: ArtifactLocation, described: str, figure: str) -> None:
        super().__init__(
            IncompleteFigureRecordsReason.CARD_DESCRIBES_OTHER_FIGURE,
            f"card {card.artifact_path} describes {described}, not {figure}",
        )
        self.card = card
        self.described = described
        self.figure = figure

    def _constructor_args(self) -> tuple[object, ...]:
        return (self.card, self.described, self.figure)


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
    """Where one figure's files and its card are, and what the figure is."""

    run_id: str
    artifact_paths: tuple[str, ...]
    card: str
    kind: FigureKind
    level: ReportLevelName
    scope: str


def index_of(cards: Mapping[ArtifactLocation, FigureCard]) -> tuple[FigureIndexEntry, ...]:
    """One entry per card, in a deterministic order, locating the files each card names."""
    entries = [
        FigureIndexEntry(
            run_id=location.run_id,
            artifact_paths=tuple(location.sibling(name).artifact_path for name in card.files),
            card=location.artifact_path,
            kind=card.metadata.kind,
            level=card.metadata.report_level,
            scope=card.scope_display,
        )
        for location, card in cards.items()
    ]
    return tuple(sorted(entries, key=lambda entry: (entry.level.value, entry.scope, entry.card)))


def _verify_table(
    figure: ArtifactLocation, card: FigureCard, uploaded: set[ArtifactLocation]
) -> None:
    match card.table:
        case SiblingTable(name=name):
            if figure.sibling(name) not in uploaded:
                raise IncompleteFigureRecordsError(
                    IncompleteFigureRecordsReason.MISSING_TABLE,
                    f"figure {figure.artifact_path} has no table",
                )
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

    A figure is complete when every one of its format files was uploaded and a card with the
    same stem sits in the same run directory and names the figure's own stem. A card without
    its figure, or an index entry whose files are not its card's, is as much a bug as a figure
    without a card.
    """
    uploaded_set = set(uploaded)
    files = {location for location in uploaded if split_figure_file(location.name) is not None}
    figures = {
        location.sibling(split[0])
        for location in files
        if (split := split_figure_file(location.name)) is not None
    }
    for figure in sorted(figures, key=lambda location: location.artifact_path):
        card_location = figure.sibling(card_name(figure.name))
        card = cards.get(card_location)
        if card is None or card_location not in uploaded_set:
            raise FigureCardMissingError(figure)
        if card.stem != figure.name:
            raise CardDescribesOtherFigureError(card_location, card.stem, figure.name)
        missing = [name for name in card.files if figure.sibling(name) not in files]
        if missing:
            raise FigureFilesMissingError(
                f"figure {figure.artifact_path} is missing files: {', '.join(missing)}",
                tuple(missing),
            )
        _verify_table(figure, card, uploaded_set)
    tables = {location for location in uploaded if location.name.endswith(TABLE_SUFFIX)}
    expected_tables = {
        card_location.sibling(table.name)
        for card_location, card in cards.items()
        if isinstance(table := card.table, SiblingTable)
    }
    if tables != expected_tables:
        raise IncompleteFigureRecordsError(
            IncompleteFigureRecordsReason.TABLES_MISMATCH,
            "the uploaded tables are not exactly the group figures'",
        )
    for card_location, card in cards.items():
        if card_location.sibling(card.stem) not in figures:
            raise IncompleteFigureRecordsError(
                IncompleteFigureRecordsReason.CARD_WITHOUT_FIGURE,
                f"card {card_location.artifact_path} has no uploaded figure {card.stem}",
            )
    indexed = [
        ArtifactLocation(entry.run_id, path) for entry in index for path in entry.artifact_paths
    ]
    if len(set(indexed)) != len(indexed) or set(indexed) != files:
        raise IncompleteFigureRecordsError(
            IncompleteFigureRecordsReason.INDEX_FIGURES_MISMATCH,
            "the figure index does not list exactly the figures",
        )
    for entry in index:
        card_location = ArtifactLocation(entry.run_id, entry.card)
        card = cards.get(card_location)
        if card is None or entry.artifact_paths != tuple(
            card_location.sibling(name).artifact_path for name in card.files
        ):
            raise IncompleteFigureRecordsError(
                IncompleteFigureRecordsReason.INDEX_ENTRY_CARD_MISMATCH,
                f"index entry names card {entry.card}, which does not describe its files "
                f"{', '.join(entry.artifact_paths)}",
            )


def index_payload(entries: Sequence[FigureIndexEntry], splits_artifact: str) -> dict[str, Any]:
    """The index as TOML-ready data; ``splits_artifact`` names the split records beside it."""
    return {
        "splits": splits_artifact,
        "figures": [
            {
                "run_id": entry.run_id,
                "artifact_paths": list(entry.artifact_paths),
                "card": entry.card,
                "kind": entry.kind.value,
                "level": entry.level.value,
                "scope": entry.scope,
            }
            for entry in entries
        ],
    }


def validate_figure_index(payload: Mapping[str, Any]) -> None:
    """Raise unless ``payload`` cites its split records and has a ``figures`` array of complete entries."""
    if not isinstance(payload.get("splits"), str):
        raise InvalidFigureIndexError(
            InvalidFigureIndexReason.NO_SPLIT_CITATION,
            "figure index does not cite its split records",
        )
    figures = payload.get("figures")
    if not isinstance(figures, list):
        raise InvalidFigureIndexError(
            InvalidFigureIndexReason.NO_FIGURES_ARRAY, "figure index has no 'figures' array"
        )
    for position, entry in enumerate(figures):
        if not isinstance(entry, Mapping):
            raise InvalidFigureIndexError(
                InvalidFigureIndexReason.ENTRY_NOT_A_TABLE,
                f"figure index entry {position} is not a table",
            )
        missing = sorted(_REQUIRED_ENTRY_FIELDS - entry.keys())
        if missing:
            raise FigureIndexEntryFieldsMissingError(
                f"figure index entry {position} is missing fields: {', '.join(missing)}",
                tuple(missing),
            )


def write_figure_index(
    path: Path, entries: Sequence[FigureIndexEntry], splits_artifact: str
) -> None:
    """Write the index as TOML, citing the split records stored in the same run."""
    path.write_bytes(tomli_w.dumps(index_payload(entries, splits_artifact)).encode("utf-8"))


def read_figure_index(path: Path) -> tuple[FigureIndexEntry, ...]:
    """Read and validate the index stored at ``path``."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_figure_index(payload)
    return tuple(
        FigureIndexEntry(
            run_id=str(entry["run_id"]),
            artifact_paths=tuple(entry["artifact_paths"]),
            card=str(entry["card"]),
            kind=FigureKind(entry["kind"]),
            level=ReportLevelName(entry["level"]),
            scope=str(entry["scope"]),
        )
        for entry in payload["figures"]
    )
