"""The figure index is derived from cards and must list exactly the uploaded figures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.figure_scope import FigureScope, RecipeScope
from neuralls.platform.reporting.figure_card import FigureCard, build_figure_card
from neuralls.platform.reporting.figure_index import (
    ArtifactLocation,
    IncompleteFigureRecordsError,
    InvalidFigureIndexError,
    index_of,
    index_payload,
    read_figure_index,
    validate_figure_index,
    verify_figure_records,
    write_figure_index,
)
from neuralls.platform.reporting.figure_metadata import FigureKind, FigureMetadata
from neuralls.shared.rhs_recipe import SinusoidalRecipe
from neuralls.shared.scientific_description.policy import NotationPolicy

DIRECTORY = "group_reports"


@pytest.fixture
def scope(dataset_presentation: DatasetPresentation) -> RecipeScope:
    return RecipeScope(dataset_presentation, SinusoidalRecipe(omega=6.28319))


@pytest.fixture
def card_of(
    metadata_of: Callable[[FigureScope], FigureMetadata],
    scope: RecipeScope,
    notation: NotationPolicy,
) -> Callable[[str], FigureCard]:
    def build(figure: str) -> FigureCard:
        return build_figure_card(
            figure, "Title", metadata_of(scope), session_run_id="session", notation=notation
        )

    return build


def _figure(run_id: str, stem: str) -> ArtifactLocation:
    return ArtifactLocation(run_id, f"{DIRECTORY}/{stem}.png")


def _card(run_id: str, stem: str) -> ArtifactLocation:
    return ArtifactLocation(run_id, f"{DIRECTORY}/{stem}.toml")


def _table(run_id: str, stem: str) -> ArtifactLocation:
    return ArtifactLocation(run_id, f"{DIRECTORY}/{stem}.parquet")


@pytest.fixture
def complete(
    card_of: Callable[[str], FigureCard],
) -> tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]]:
    """Two group figures, one in each of two runs, sharing a stem, each with card and table."""
    uploaded = [
        location
        for run_id in ("root", "child")
        for location in (
            _figure(run_id, "residual"),
            _card(run_id, "residual"),
            _table(run_id, "residual"),
        )
    ]
    cards = {_card(run_id, "residual"): card_of("residual.png") for run_id in ("root", "child")}
    return uploaded, cards


def test_index_lists_every_card_with_where_it_is_and_what_it_shows(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    _, cards = complete

    entries = index_of(cards)

    assert {(e.run_id, e.artifact_path, e.card) for e in entries} == {
        (run_id, f"{DIRECTORY}/residual.png", f"{DIRECTORY}/residual.toml")
        for run_id in ("root", "child")
    }
    assert {e.kind for e in entries} == {card.metadata.kind for card in cards.values()}


def test_complete_records_pass_verification(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete

    verify_figure_records(uploaded, cards, index_of(cards))


def test_a_figure_without_a_card_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete
    uploaded.append(_figure("root", "orphan"))

    with pytest.raises(IncompleteFigureRecordsError, match="orphan.png has no card"):
        verify_figure_records(uploaded, cards, index_of(cards))


def test_a_group_figure_without_its_table_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete
    uploaded.remove(_table("child", "residual"))

    with pytest.raises(IncompleteFigureRecordsError, match="has no table"):
        verify_figure_records(uploaded, cards, index_of(cards))


def test_a_table_beside_a_history_figure_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
    card_of: Callable[[str], FigureCard],
) -> None:
    uploaded, cards = complete
    history = card_of("residual.png")
    cards[_card("root", "residual")] = replace(
        history, metadata=replace(history.metadata, kind=FigureKind.RESIDUAL_HISTORY)
    )

    with pytest.raises(IncompleteFigureRecordsError, match="tables"):
        verify_figure_records(uploaded, cards, index_of(cards))


def test_a_card_naming_another_file_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
    card_of: Callable[[str], FigureCard],
) -> None:
    uploaded, cards = complete
    cards[_card("root", "residual")] = card_of("energy.png")

    with pytest.raises(IncompleteFigureRecordsError, match="describes energy.png"):
        verify_figure_records(uploaded, cards, index_of(cards))


def test_a_card_without_its_figure_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete
    uploaded.remove(_figure("child", "residual"))

    with pytest.raises(IncompleteFigureRecordsError, match="no uploaded figure"):
        verify_figure_records(uploaded, cards, index_of(cards))


def test_an_index_entry_without_a_figure_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete
    ghost = replace(index_of(cards)[0], artifact_path=f"{DIRECTORY}/ghost.png")

    with pytest.raises(IncompleteFigureRecordsError, match="exactly the figures"):
        verify_figure_records(uploaded, cards, [*index_of(cards), ghost])


def test_a_figure_missing_from_the_index_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete

    with pytest.raises(IncompleteFigureRecordsError, match="exactly the figures"):
        verify_figure_records(uploaded, cards, index_of(cards)[:1])


def test_an_index_entry_naming_a_different_card_is_rejected(
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    uploaded, cards = complete
    entries = [replace(entry, card=f"{DIRECTORY}/other.toml") for entry in index_of(cards)]

    with pytest.raises(IncompleteFigureRecordsError, match="names card"):
        verify_figure_records(uploaded, cards, entries)


def test_index_round_trips_through_toml_without_card_parameters(
    tmp_path: Path,
    complete: tuple[list[ArtifactLocation], dict[ArtifactLocation, FigureCard]],
) -> None:
    _, cards = complete
    entries = index_of(cards)
    path = tmp_path / "figure_index.toml"

    write_figure_index(path, entries)

    assert read_figure_index(path) == entries
    assert all(
        set(entry) == {"run_id", "artifact_path", "card", "kind", "level", "scope"}
        for entry in index_payload(entries)["figures"]
    )


def test_index_entry_missing_a_field_is_rejected() -> None:
    with pytest.raises(InvalidFigureIndexError, match="figures"):
        validate_figure_index({})
    with pytest.raises(InvalidFigureIndexError, match="card"):
        validate_figure_index(
            {
                "figures": [
                    {
                        "run_id": "r",
                        "artifact_path": "a.png",
                        "kind": "k",
                        "level": "l",
                        "scope": "s",
                    }
                ]
            }
        )
