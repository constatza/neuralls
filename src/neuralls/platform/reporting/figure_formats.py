"""The file formats every report figure is written in, and where those files go.

A figure is one picture saved once per ``FigureFormat``; today that is the PNG a browser shows.
This module is the only place that lists the formats; the writer, the card, the index and the
uploader all derive the file names from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FigureFormat(StrEnum):
    """A format a figure is written in; the value is the file suffix without its dot."""

    PNG = "png"


FIGURE_FORMATS: tuple[FigureFormat, ...] = tuple(FigureFormat)
"""Every format a figure is written in, in the order its files are listed."""


def figure_file(stem: str, figure_format: FigureFormat) -> str:
    """The file name of the ``figure_format`` file of the figure with file stem ``stem``."""
    return f"{stem}.{figure_format.value}"


def figure_files(stem: str) -> tuple[str, ...]:
    """The file names of one figure, one per format."""
    return tuple(figure_file(stem, figure_format) for figure_format in FIGURE_FORMATS)


def split_figure_file(name: str) -> tuple[str, FigureFormat] | None:
    """The stem and format of a figure file ``name``, or ``None`` for any other file."""
    stem, dot, suffix = name.rpartition(".")
    if not dot or not stem:
        return None
    try:
        return stem, FigureFormat(suffix)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class FigureTarget:
    """Where one figure is written: a directory and the file stem shared by its format files."""

    directory: Path
    stem: str

    def __post_init__(self) -> None:
        """Require a stem that names a file, not a path."""
        if not self.stem or Path(self.stem).name != self.stem:
            raise ValueError(f"figure stem must be a file name without directories: {self.stem!r}")

    def path(self, figure_format: FigureFormat) -> Path:
        """The path of this figure's ``figure_format`` file."""
        return self.directory / figure_file(self.stem, figure_format)

    @property
    def paths(self) -> tuple[Path, ...]:
        """The path of every file of this figure, one per format."""
        return tuple(self.path(figure_format) for figure_format in FIGURE_FORMATS)
