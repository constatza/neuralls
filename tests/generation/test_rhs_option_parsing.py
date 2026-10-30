"""The weakly-typed ``provide_rhs`` value parses once into a closed set of RHS options."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation._archive_resolution import (
    NoRhs,
    RhsFromFile,
    RhsFromSolutions,
    RhsOption,
    parse_provide_rhs,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, NoRhs()),
        (False, NoRhs()),
        (True, RhsFromSolutions()),
        ("rhs.txt", RhsFromFile(Path("rhs.txt"))),
        (Path("dir/rhs.txt"), RhsFromFile(Path("dir/rhs.txt"))),
    ],
)
def test_parses_each_value_shape(value: bool | str | Path | None, expected: RhsOption) -> None:
    assert parse_provide_rhs(value) == expected
