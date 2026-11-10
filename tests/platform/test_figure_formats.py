"""The written figure formats are defined once; the file names and the writer derive from them."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pytest

from neuralls.platform.reporting.figure_formats import (
    FIGURE_FORMATS,
    FigureFormat,
    FigureTarget,
    figure_files,
    split_figure_file,
)
from neuralls.platform.reporting.figure_writer import finish_group_figure


def test_a_figure_is_written_as_a_raster_file() -> None:
    assert FIGURE_FORMATS == (FigureFormat.PNG,)


def test_a_stem_names_one_file_per_format() -> None:
    assert figure_files("residual.v2") == ("residual.v2.png",)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("residual.v2.png", ("residual.v2", FigureFormat.PNG)),
        ("residual.pdf", None),
        ("residual.png", ("residual", FigureFormat.PNG)),
        ("residual.parquet", None),
        ("residual.toml", None),
        ("png", None),
        (".png", None),
    ],
)
def test_a_file_name_is_split_into_stem_and_format_or_is_not_a_figure_file(
    name: str, expected: tuple[str, FigureFormat] | None
) -> None:
    assert split_figure_file(name) == expected


def test_a_target_has_one_path_per_format(tmp_path: Path) -> None:
    target = FigureTarget(tmp_path, "residual")

    assert target.paths == (tmp_path / "residual.png",)


@pytest.mark.parametrize("stem", ["", "nested/figure"])
def test_a_target_stem_names_a_file_not_a_path(tmp_path: Path, stem: str) -> None:
    with pytest.raises(ValueError):
        FigureTarget(tmp_path, stem)


def test_the_figure_writer_saves_every_format_of_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[object] = []

    def record(
        self: matplotlib.figure.Figure, fname: object, *args: object, **kwargs: object
    ) -> None:
        saved.append(fname)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record)
    target = FigureTarget(tmp_path / "nested", "residual")
    fig = matplotlib.figure.Figure()

    finish_group_figure(
        fig,
        fig.subplots(),
        title="Title",
        subtitle=None,
        target=target,
        show=False,
        saved_message="Saved",
    )

    assert saved == list(target.paths)
    assert target.directory.is_dir()
