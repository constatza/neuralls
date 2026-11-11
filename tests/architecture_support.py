"""One parse per source file for the source lints that walk the same trees."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path


@cache
def parsed(path: Path) -> ast.Module:
    """The syntax tree of `path`, parsed once per process however many lints read it."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
