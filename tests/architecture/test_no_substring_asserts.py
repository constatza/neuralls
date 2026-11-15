"""Tests compare text as whole values, never by a substring of it.

A substring assert passes on text that gained or lost everything around the substring, so a
rendered label, message or file must be compared in full (or parsed and compared as data).

shortcut: name-based heuristic over the asserted expression, upgrade if a substring assert
slips through under another name.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.architecture_support import parsed

_TESTS = Path(__file__).resolve().parents[1]
_TEXT_CALLS = frozenset({"str", "repr", "format", "read_text", "getvalue"})
_TEXT_NAMES = frozenset(
    {
        "text", "msg", "message", "output", "stdout", "stderr",
        "name", "label", "title", "summary", "value",
    }
)  # fmt: skip


def _last_identifier(node: ast.expr) -> str | None:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return None


def _is_text(node: ast.expr) -> bool:
    match node:
        case ast.Call(func=func):
            return _last_identifier(func) in _TEXT_CALLS
        case _:
            return _last_identifier(node) in _TEXT_NAMES


def _substring_asserts(path: Path, *, root: Path = _TESTS) -> list[str]:
    found = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Assert(
                test=ast.Compare(
                    left=ast.Constant(value=str()), ops=[ast.In() | ast.NotIn()], comparators=[text]
                )
            ) if _is_text(text):
                found.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    return found


def test_no_test_asserts_a_substring_of_text() -> None:
    offenders = [
        location for path in sorted(_TESTS.rglob("*.py")) for location in _substring_asserts(path)
    ]

    assert offenders == []


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """Three substring asserts on text (lines 1-3), then a full-value one and a membership one."""
    path = tmp_path / "test_planted.py"
    path.write_text(
        'assert "a" in label\n'
        'assert "a" not in str(error)\n'
        'assert "a" in record.message\n'
        'assert label == "a"\n'
        'assert "a" in keys\n',
        encoding="utf-8",
    )
    return path


def test_the_lint_flags_substring_asserts_on_text_and_nothing_else(
    planted: Path, tmp_path: Path
) -> None:
    assert _substring_asserts(planted, root=tmp_path) == [
        "test_planted.py:1",
        "test_planted.py:2",
        "test_planted.py:3",
    ]
