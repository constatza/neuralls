"""Shared JSON coercion for reporting payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_json_primitive(value: Any) -> Any:
    """Coerce a reporting payload value into JSON-serializable primitives.

    The single coercion used by every reporting writer, so a comparison
    artifact and its staged input provenance agree on how the same value is
    rendered. ``Enum`` is matched ahead of the primitive case on purpose:
    ``RowKind`` is an ``int`` enum with a custom ``__str__``, and must
    serialize as its lowercase name rather than its integer value.

    Args:
        value: Any reporting payload value — primitive, ``Path``, ``Enum``,
            mapping, sequence, or (possibly nested) dataclass instance.

    Returns:
        An equivalent value built only from ``None``/``bool``/``int``/
        ``float``/``str``/``list``/``dict``.

    Raises:
        TypeError: If the value has no JSON representation.
    """
    match value:
        case Path() as path:
            return path.as_posix()
        case Enum() as member:
            return str(member)
        case None | bool() | int() | float() | str():
            return value
        case dict() as mapping:
            return {str(key): to_json_primitive(item) for key, item in mapping.items()}
        case list() | tuple() as sequence:
            return [to_json_primitive(item) for item in sequence]
        case _ if is_dataclass(value) and not isinstance(value, type):
            return to_json_primitive(asdict(value))
        case _:
            raise TypeError(f"Unsupported reporting payload value: {type(value).__name__}")
