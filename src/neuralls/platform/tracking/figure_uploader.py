"""Staging and upload of report figures with their cards, tables and the figure index."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, assert_never

import mlflow

from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.platform.reporting.figure_card import (
    FigureCard,
    build_figure_card,
    card_name,
    write_figure_card,
)
from neuralls.platform.reporting.figure_index import (
    FIGURE_INDEX_FILE,
    ArtifactLocation,
    FigureIndexEntry,
    index_of,
    verify_figure_records,
    write_figure_index,
)
from neuralls.platform.reporting.figure_metadata import (
    NoTable,
    SiblingTable,
    StagedFigure,
    StagedTabledFigure,
    UploadableFigure,
    collision_key,
)
from neuralls.platform.reporting.figure_table import check_table_kind, write_group_table
from neuralls.platform.tracking.split_records import SPLITS_ARTIFACT, encode_split_records
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import SessionRunId


class DuplicateFigureNameError(ValueError):
    """Two figures of one run directory would share a file name once made filesystem-safe."""


class ArtifactSink(Protocol):
    """Logs a local file into the run that is currently active; the only MLflow call needed."""

    def log(self, local_path: Path, directory: str | None) -> None:
        """Upload ``local_path`` into ``directory`` of the active run's artifacts."""
        ...


@dataclass(frozen=True, slots=True)
class MlflowArtifactSink:
    """Logs artifacts through the active MLflow run."""

    def log(self, local_path: Path, directory: str | None) -> None:
        """Upload ``local_path`` into ``directory`` of the active run's artifacts."""
        mlflow.log_artifact(str(local_path), artifact_path=directory)


@dataclass(slots=True)
class FigureUploader:
    """Uploads each figure's files with its card (and plotted table) and records what was uploaded.

    Figures are staged in a private temporary directory. File stems only need to be unique
    inside one run directory; the uploader rejects a name that collides with another one in
    the same ``(run, directory)``. ``write_index`` checks that every figure has its card and
    uploads the index of all figures into the active (root) run.
    """

    session_run_id: SessionRunId
    notation: NotationPolicy
    artifacts: ArtifactSink = field(default_factory=MlflowArtifactSink)
    _staging: tempfile.TemporaryDirectory[str] = field(
        default_factory=lambda: tempfile.TemporaryDirectory(prefix="neuralls-figures-")
    )
    _taken: dict[tuple[str, str | None], set[str]] = field(default_factory=dict)
    _uploaded: list[ArtifactLocation] = field(default_factory=list)
    _cards: dict[ArtifactLocation, FigureCard] = field(default_factory=dict)

    def staging_dir(self, name: str) -> Path:
        """A directory called ``name`` inside the staging area, created on first request."""
        path = Path(self._staging.name) / name
        path.mkdir(exist_ok=True)
        return path

    def check_free(self, run_id: str, directory: str | None, stems: Iterable[str]) -> None:
        """Reject stems that collide with each other or with a figure already in the directory."""
        seen = set(self._taken.get((run_id, directory), ()))
        for stem in stems:
            key = collision_key(stem)
            if key in seen:
                raise DuplicateFigureNameError(f"duplicate or colliding figure stem: {stem}")
            seen.add(key)

    def upload(self, run_id: str, directory: str | None, figure: UploadableFigure) -> None:
        """Upload a figure, its card and, for a tabled figure, its plotted table side by side.

        The card decides whether the figure has a table; a mismatch with the figure's type
        is a caller error.
        """
        match figure:
            case StagedFigure() as staged:
                card = self._upload_with_card(run_id, directory, staged)
                match card.table:
                    case NoTable():
                        pass
                    case SiblingTable():
                        raise ValueError(f"figure {staged.plan.stem} needs its plotted table")
                    case unreachable:
                        assert_never(unreachable)
            case StagedTabledFigure(figure=staged, table=table):
                card = self._upload_with_card(run_id, directory, staged)
                match card.table:
                    case SiblingTable(name=name, kind=kind):
                        table_path = staged.target.directory / name
                        write_group_table(table_path, check_table_kind(table, kind))
                        self.artifacts.log(table_path, directory)
                        self._uploaded.append(ArtifactLocation(run_id, _in(directory, name)))
                    case NoTable():
                        raise ValueError(f"figure {staged.plan.stem} has no plotted table")
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)

    def _upload_with_card(
        self, run_id: str, directory: str | None, staged: StagedFigure
    ) -> FigureCard:
        stem = staged.plan.stem
        self.check_free(run_id, directory, [stem])
        card = build_figure_card(
            stem,
            staged.plan.title,
            staged.plan.metadata,
            session_run_id=self.session_run_id,
            notation=self.notation,
        )
        card_path = staged.target.directory / card_name(stem)
        write_figure_card(card_path, card)
        for path in staged.target.paths:
            self.artifacts.log(path, directory)
        self.artifacts.log(card_path, directory)
        self._taken.setdefault((run_id, directory), set()).add(collision_key(stem))
        card_location = ArtifactLocation(run_id, _in(directory, card_path.name))
        self._uploaded.extend(ArtifactLocation(run_id, _in(directory, name)) for name in card.files)
        self._uploaded.append(card_location)
        self._cards[card_location] = card
        return card

    def write_index(
        self, split_records: Sequence[DatasetSplitRecord]
    ) -> tuple[FigureIndexEntry, ...]:
        """Check every figure has its card, upload the split records and the index that cites them.

        Both land in the active (root) run, so the report carries what its session held out.
        """
        index = index_of(self._cards)
        verify_figure_records(self._uploaded, self._cards, index)
        staging = Path(self._staging.name)
        splits_path = staging / SPLITS_ARTIFACT
        splits_path.write_bytes(encode_split_records(split_records))
        index_path = staging / FIGURE_INDEX_FILE
        write_figure_index(index_path, index, SPLITS_ARTIFACT)
        self.artifacts.log(splits_path, None)
        self.artifacts.log(index_path, None)
        self._staging.cleanup()
        return index


def _in(directory: str | None, name: str) -> str:
    """The artifact path of ``name`` inside ``directory`` (the run root when ``None``)."""
    return f"{directory}/{name}" if directory else name
