"""Windows defaults text I/O to cp1252, so every text read or write must name its encoding."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("src", "tests", "scripts")
_PATH_TEXT_METHODS = {"read_text", "write_text"}
_BINARY_MODE_FLAG = "b"


def _is_binary_open(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else None
    mode = next((k.value for k in call.keywords if k.arg == "mode"), mode)
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and _BINARY_MODE_FLAG in mode.value
    )


def _is_unencoded_text_io(call: ast.Call) -> bool:
    if any(keyword.arg == "encoding" for keyword in call.keywords):
        return False
    match call.func:
        case ast.Attribute(attr=attr) if attr in _PATH_TEXT_METHODS:
            return True
        case ast.Name(id="open") | ast.Attribute(value=ast.Name(id="io"), attr="open"):
            return not _is_binary_open(call)
        case _:
            return False


def test_text_io_always_names_its_encoding() -> None:
    offenders = [
        f"{source.relative_to(_REPO)}:{node.lineno}"
        for root in _ROOTS
        for source in sorted((_REPO / root).rglob("*.py"))
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and _is_unencoded_text_io(node)
    ]
    assert offenders == []
