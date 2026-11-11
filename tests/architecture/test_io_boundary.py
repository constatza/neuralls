"""Real IO belongs to the IO-boundary test directories; elsewhere it is declared and temporary.

A test file outside the boundary that creates files, a dataset, or an MLflow store must carry
``pytestmark = pytest.mark.deferred_io("<reason>")`` naming the future refactor that will
remove that IO: ``mlflow-run-tracking`` (MLflow runs behind ports), ``dataset-fixture-io`` (a test
that needs a dataset directory on disk as its input fixture) or ``digest-hashing`` (file hashing behind a port). A file
that no longer does IO must drop the marker, so the set only shrinks.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture_support import parsed

_TESTS = Path(__file__).resolve().parents[1]
_BOUNDARY_DIRS = ("io", "integration", "platform", "architecture", "manual")
_REASONS = frozenset({"mlflow-run-tracking", "dataset-fixture-io", "digest-hashing"})
_PURE_LAYERS = ("domain", "application")
_SRC = _TESTS.parent / "src" / "neuralls"
_FILESYSTEM_METHODS = frozenset(
    {
        "exists", "is_file", "is_dir", "glob", "rglob", "iterdir", "stat", "mkdir", "open",
        "read_text", "write_text", "read_bytes", "write_bytes", "unlink",
    }
)  # fmt: skip
_ARCHIVE_SOURCE_NAMES = frozenset({"source", "archive_source"})
_IO_USE = re.compile(
    r"(?<![.\w])(tmp_path|tmp_path_factory|mlflow_tracking_uri|mlflow_experiment|"
    r"log_run_with_checkpoint|log_run_with_named_checkpoints|build_test_dataset)\b"
    r"|\.write_text\(|\.write_bytes\(|np\.save|mmwrite|tomli_w"
)


def _test_files_outside_the_boundary() -> list[Path]:
    return [
        path
        for path in sorted(_TESTS.rglob("test_*.py"))
        if path.relative_to(_TESTS).parts[0] not in _BOUNDARY_DIRS
    ]


def _deferred_plan(source: str) -> str | None:
    """The plan named by a module-level ``pytestmark = pytest.mark.deferred_io("Dn")``."""
    for node in ast.parse(source).body:
        match node:
            case ast.Assign(
                targets=[ast.Name(id="pytestmark")],
                value=ast.Call(
                    func=ast.Attribute(attr="deferred_io"),
                    args=[ast.Constant(value=str(plan))],
                ),
            ):
                return plan
    return None


def test_io_outside_the_boundary_is_declared_with_its_retiring_plan() -> None:
    undeclared, unknown_plan, stale = [], [], []
    for path in _test_files_outside_the_boundary():
        source = path.read_text(encoding="utf-8")
        plan = _deferred_plan(source)
        name = path.relative_to(_TESTS).as_posix()
        uses_io = _IO_USE.search(source) is not None
        if uses_io and plan is None:
            undeclared.append(name)
        elif plan is not None and plan not in _REASONS:
            unknown_plan.append(name)
        elif plan is not None and not uses_io:
            stale.append(name)

    assert (undeclared, unknown_plan, stale) == ([], [], [])


def _filesystem_calls(path: Path) -> list[int]:
    """Lines of file I/O calls; the injected archive source port is the only allowed receiver."""
    lines = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Call(func=ast.Name(id="open")):
                lines.append(node.lineno)
            case ast.Call(func=ast.Attribute(attr=attr, value=ast.Name(id=receiver))) if (
                attr in _FILESYSTEM_METHODS and receiver not in _ARCHIVE_SOURCE_NAMES
            ) or (receiver in {"np", "numpy"} and attr.startswith(("load", "save"))):
                lines.append(node.lineno)
            case ast.Call(func=ast.Attribute(attr=attr, value=receiver)) if (
                attr in _FILESYSTEM_METHODS and not isinstance(receiver, ast.Name)
            ):
                lines.append(node.lineno)
    return lines


def test_pure_layers_do_no_file_io() -> None:
    offenders = [
        f"{path.relative_to(_SRC).as_posix()}:{line}"
        for layer in _PURE_LAYERS
        for path in sorted((_SRC / layer).rglob("*.py"))
        for line in _filesystem_calls(path)
    ]
    assert offenders == []
