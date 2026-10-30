"""Timed regions measure the solver, so no persistence or tracking call may run inside one."""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "neuralls"
_TIMERS = {
    "repeat_timed",
    "repeat_timed_outcome",
    "track_resource_usage",
    "begin_resource_usage",
}
_IO_CALLS = {
    "write_parquet",
    "write_table",
    "savefig",
    "log_artifact",
    "log_metric",
    "log_param",
    "write_text",
    "write_bytes",
    "open",
    "dump",
    "dumps",
}


def _called_name(call: ast.Call) -> str | None:
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return None


def _io_calls_in_timer_arguments(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node) not in _TIMERS:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        lines.extend(
            inner.lineno
            for argument in arguments
            for inner in ast.walk(argument)
            if isinstance(inner, ast.Call) and _called_name(inner) in _IO_CALLS
        )
    return lines


def test_no_io_call_inside_a_timer_argument() -> None:
    offenders = [
        f"{source.relative_to(_SOURCE)}:{line}"
        for source in sorted(_SOURCE.rglob("*.py"))
        for line in _io_calls_in_timer_arguments(ast.parse(source.read_text(encoding="utf-8")))
    ]
    assert offenders == []


def test_the_detector_flags_io_inside_a_lambda_argument() -> None:
    tree = ast.parse("repeat_timed(lambda: frame.write_parquet(path))")
    assert _io_calls_in_timer_arguments(tree) == [1]
