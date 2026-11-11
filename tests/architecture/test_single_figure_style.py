"""Matplotlib style state is changed only in `figure_style.py`, so figures share one look."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture_support import parsed

_REPO = Path(__file__).resolve().parents[2]
_STYLE_MODULE = _REPO / "src/neuralls/platform/reporting/figure_style.py"
_STYLE_CALLS = frozenset({"rc_context", "set_theme"})


def _is_style_use(node: ast.Attribute) -> bool:
    return (
        node.attr == "use" and isinstance(node.value, ast.Attribute) and node.value.attr == "style"
    )


def _style_lines(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and (node.attr in _STYLE_CALLS or _is_style_use(node))
    ]


def test_style_context_appears_only_in_the_figure_style_module() -> None:
    offenders = [
        f"{source.relative_to(_REPO)}:{line}"
        for source in sorted((_REPO / "src").rglob("*.py"))
        if source != _STYLE_MODULE
        for line in _style_lines(parsed(source))
    ]
    assert offenders == []
