"""Source lint: compare and report share only stored artifacts, never in-memory types."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path(sys.modules["neuralls"].__file__).parent
POSTPROCESSING = "neuralls.composition.comparison.postprocessing"
REPORT_ENTRY = PACKAGE / "composition" / "assignments" / "comparison_report.py"
FORBIDDEN_COMPUTE_TYPES = frozenset({"CompletedMethodWorkload", "CGComparisonResult"})


def _imported(path: Path) -> list[tuple[str, str | None]]:
    """(module, imported name) for every import statement; name is None for plain imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found += [(node.module, alias.name) for alias in node.names]
    return found


def test_compute_modules_never_import_postprocessing() -> None:
    postprocessing_dir = PACKAGE / "composition" / "comparison" / "postprocessing"
    offenders = [
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if postprocessing_dir not in path.parents
        and path not in {REPORT_ENTRY, PACKAGE / "composition" / "assignments" / "case_pipeline.py"}
        and path != PACKAGE / "cli" / "report.py"
        and any(module.startswith(POSTPROCESSING) for module, _ in _imported(path))
    ]
    assert offenders == []


def test_postprocessing_never_imports_compute_time_types() -> None:
    postprocessing_dir = PACKAGE / "composition" / "comparison" / "postprocessing"
    offenders = [
        f"{path.name}: {name}"
        for path in postprocessing_dir.glob("*.py")
        for _, name in _imported(path)
        if name in FORBIDDEN_COMPUTE_TYPES
    ]
    assert offenders == []
