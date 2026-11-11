"""Every report figure is written by `finish_group_figure`, so an output-format change is one edit."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture_support import parsed

_REPO = Path(__file__).resolve().parents[2]
_WRITER = "finish_group_figure"
_WRITER_MODULE = _REPO / "src/neuralls/platform/reporting/figure_writer.py"


def _savefig_lines(tree: ast.AST, *, is_writer_module: bool) -> list[int]:
    """Lines of `.savefig` accesses outside the writer function of the writer module."""
    writer_nodes = {
        id(node)
        for function in ast.walk(tree)
        if is_writer_module and isinstance(function, ast.FunctionDef) and function.name == _WRITER
        for node in ast.walk(function)
    }
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "savefig"
        and id(node) not in writer_nodes
    ]


def test_savefig_appears_only_inside_the_figure_writer() -> None:
    offenders = [
        f"{source.relative_to(_REPO)}:{line}"
        for source in sorted((_REPO / "src").rglob("*.py"))
        for line in _savefig_lines(parsed(source), is_writer_module=source == _WRITER_MODULE)
    ]
    assert offenders == []
