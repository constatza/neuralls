"""Characterization tests for the shared reporting JSON coercion helper.

`platform/reporting/artifacts.py` and `platform/reporting/comparison_inputs.py`
each used to carry their own near-identical dataclass/Path/Enum/dict/list
coercion. These pin the union of both behaviors that the merged helper must
keep — in particular `RowKind`'s custom ``__str__`` (an int enum that must
serialize as its lowercase name, not its integer value).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from neuralls.platform.reporting.serialization import to_json_primitive
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


@dataclass(frozen=True)
class _Leaf:
    """A nested dataclass exercising recursive coercion."""

    path: Path
    count: int


@dataclass(frozen=True)
class _Node:
    """A dataclass holding another dataclass plus a collection."""

    leaf: _Leaf
    labels: tuple[str, ...]


@pytest.fixture
def nested_payload() -> _Node:
    """A dataclass tree with a Path leaf and a tuple of labels."""
    return _Node(leaf=_Leaf(path=Path("figures/convergence.png"), count=3), labels=("a", "b"))


@pytest.mark.parametrize("value", [None, True, False, 0, 3, 1.5, "text"])
def test_json_primitives_pass_through_unchanged(value: object) -> None:
    assert to_json_primitive(value) == value


def test_paths_become_posix_strings() -> None:
    assert to_json_primitive(Path("figures") / "convergence.png") == "figures/convergence.png"


def test_str_enums_serialize_as_their_value() -> None:
    assert to_json_primitive(ComparisonRhsSourceKind.RAW_LHS) == "raw_lhs"


def test_int_enums_serialize_via_their_own_str() -> None:
    """RowKind is an int enum with a custom __str__ — its name, never its int value."""
    assert to_json_primitive(RowKind.CG_INTERNAL) == "cg_internal"


def test_dict_keys_are_stringified_and_values_coerced() -> None:
    assert to_json_primitive({1: Path("a.npy"), "b": RowKind.STANDARD}) == {
        "1": "a.npy",
        "b": "standard",
    }


def test_lists_and_tuples_both_become_lists() -> None:
    assert to_json_primitive((Path("a"), [Path("b")])) == ["a", ["b"]]


def test_dataclasses_are_recursively_flattened(nested_payload: _Node) -> None:
    assert to_json_primitive(nested_payload) == {
        "leaf": {"path": "figures/convergence.png", "count": 3},
        "labels": ["a", "b"],
    }


def test_unsupported_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        to_json_primitive(object())
