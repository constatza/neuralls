"""Source lint: production code builds scientific facts only through the catalog factory."""

from __future__ import annotations

import ast
from pathlib import Path

import neuralls

_FACT_CONSTRUCTORS = frozenset({"IntegerFact", "DecimalFact", "TextFact", "DisplayTextFact"})


def test_production_code_builds_facts_only_through_the_catalog_factory() -> None:
    """The render-time category and kind checks stay a backstop, never a reachable path."""
    package = Path(neuralls.__file__).parent
    allowed = {package / "shared" / "scientific_description" / "catalog.py"}
    offenders: list[str] = []
    for source in package.rglob("*.py"):
        if source in allowed:
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _FACT_CONSTRUCTORS
            ):
                offenders.append(f"{source.relative_to(package)}:{node.lineno}")
    assert offenders == []
