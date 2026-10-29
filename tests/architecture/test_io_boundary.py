"""Real IO belongs to the IO-boundary test directories; elsewhere it is declared and temporary.

A test file outside the boundary that creates files, a dataset, or an MLflow store must carry
``pytestmark = pytest.mark.deferred_io("<reason>")`` naming the future refactor that will
remove that IO: ``mlflow-run-tracking`` (MLflow runs behind ports), ``generation-io`` (dataset
generation file access behind ports), ``config-loading`` (config load/bind split) or
``digest-hashing`` (file hashing behind a port). A file
that no longer does IO must drop the marker, so the set only shrinks.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_BOUNDARY_DIRS = ("io", "integration", "platform", "architecture", "manual")
_REASONS = frozenset({"mlflow-run-tracking", "generation-io", "config-loading", "digest-hashing"})
_IO_USE = re.compile(
    r"\b(tmp_path|tmp_path_factory|mlflow_tracking_uri|mlflow_experiment|"
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
