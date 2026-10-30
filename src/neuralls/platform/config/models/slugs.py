"""Slug-safe text rendering shared by the preconditioner config models."""

from __future__ import annotations


def slug_number(value: float) -> str:
    """Slug-safe rendering of a float: no ``.``/``-``/``+`` (e.g. a ``%g``-formatted exponent)."""
    return f"{value:g}".replace(".", "p").replace("-", "neg").replace("+", "")


def slug_text(text: str) -> str:
    """Lowercase, alnum-and-underscore-only rendering of free text from a checkpoint ref."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_") or "unset"
