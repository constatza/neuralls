"""Source lint: the scientific_description package imports only the standard library."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def test_package_imports_standard_library_only() -> None:
    """No module under ``scientific_description`` imports outside the stdlib."""
    package_dir = (
        Path(sys.modules["neuralls"].__file__).parent / "shared" / "scientific_description"
    )
    stdlib_names = set(sys.stdlib_module_names) | {"neuralls"}
    for source_file in package_dir.glob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                assert name in stdlib_names, f"{source_file.name} imports non-stdlib {name!r}"
