"""Deterministic, content-addressed digests of configs and files.

Pure helpers with no domain knowledge. ``canonical`` maps a value to a
JSON-able structure (failing fast on anything it does not understand) and
``canonical_digest`` hashes that structure. Field-level markers used inside
``Annotated[...]`` select how a model field contributes to identity:

* ``Cosmetic``: excluded (labels, output locations).
* ``InputData``: a ``Path`` hashed by file/directory bytes.
* ``InputConfig``: a ``Path`` to a TOML file hashed by its parsed meaning.
* no marker: hashed by value (default-include).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from pathlib import Path, PurePath
from typing import Annotated, Any, Protocol, get_args, get_origin, get_type_hints

import numpy as np
from pydantic import BaseModel

type Digest = str
"""Always ``"sha256:<64 hex>"``."""

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]

_ALGORITHM = "sha256"


@dataclass(frozen=True)
class Cosmetic:
    """Marker: the field is excluded from identity (label or location)."""


@dataclass(frozen=True)
class InputData:
    """Marker: a ``Path`` field hashed as bytes via ``content_digest``."""


@dataclass(frozen=True)
class InputConfig:
    """Marker: a ``Path`` field hashed semantically via ``toml_digest``."""


type Marker = Cosmetic | InputData | InputConfig
MARKER_TYPES: tuple[type, ...] = (Cosmetic, InputData, InputConfig)


def _dumps(structure: Json) -> str:
    """Serialise a canonical structure to its stable JSON text.

    Args:
        structure: Output of ``canonical``.

    Returns:
        Compact, key-sorted, ASCII JSON text.

    Raises:
        ValueError: If the structure contains NaN or infinity.
    """
    return json.dumps(
        structure, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    )


_ARRAY_CHUNK_BYTES = 64 * 1024 * 1024


class RowSliceable(Protocol):
    """Array-like readable in axis-0 slices (numpy, memmap, h5py, zarr)."""

    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Any) -> Any:
        """Return the sub-array selected by ``key`` (a row slice, or ``()`` for 0-d)."""
        ...


def array_digest(array: RowSliceable) -> Digest:
    """Digest an array's *logical* content: dtype, shape and values.

    Independent of the storage format (npy/HDF5/zarr) and of library-version
    file layout. Values stream through in axis-0 chunks, so an on-disk array
    is never fully materialized; byte order is normalized to little-endian.

    Args:
        array (RowSliceable): Array-like exposing ``shape``, ``dtype`` and
            axis-0 slicing.

    Returns:
        Digest: ``sha256:<hex>`` over the header and all values.
    """
    dtype = np.dtype(array.dtype).newbyteorder("<")
    shape = tuple(int(n) for n in array.shape)
    hasher = hashlib.sha256(f"{dtype.str}|{shape}".encode("ascii"))
    if not shape:
        hasher.update(np.asarray(array[()], dtype=dtype).tobytes())
        return _format(hasher.hexdigest())
    row_bytes = max(1, dtype.itemsize * math.prod(shape[1:]))
    rows_per_chunk = max(1, _ARRAY_CHUNK_BYTES // row_bytes)
    for start in range(0, shape[0], rows_per_chunk):
        chunk = np.asarray(array[start : start + rows_per_chunk], dtype=dtype)
        hasher.update(np.ascontiguousarray(chunk).tobytes())
    return _format(hasher.hexdigest())


def _format(hexdigest: str) -> Digest:
    return f"{_ALGORITHM}:{hexdigest}"


def canonical_digest(*parts: object) -> Digest:
    """Digest the canonical JSON of ``parts``.

    Args:
        *parts: Any values understood by ``canonical``.

    Returns:
        ``"sha256:<hex>"`` digest.

    Raises:
        TypeError: If a part has an unsupported type.
        ValueError: If a part contains a non-finite float.
    """
    text = _dumps(canonical(list(parts)))
    return _format(hashlib.sha256(text.encode("ascii")).hexdigest())


@singledispatch
def canonical(obj: object) -> Json:
    """Return a JSON-able canonical form of ``obj``.

    Dataclass instances are handled here (they share no base class); every
    other supported type has a registered implementation.

    Args:
        obj: Value to canonicalize.

    Returns:
        JSON-able structure.

    Raises:
        TypeError: If the type is not supported (there is no ``str()`` fallback).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        hints = get_type_hints(type(obj), include_extras=True)
        names = [f.name for f in dataclasses.fields(obj)]
        return _fields(obj, names, {n: _annotated_markers(hints[n]) for n in names})
    raise TypeError(f"Cannot canonicalize {type(obj).__module__}.{type(obj).__qualname__}")


@canonical.register
def _(obj: None) -> Json:
    return None


@canonical.register
def _(obj: bool) -> Json:
    return obj


@canonical.register
def _(obj: int) -> Json:
    return int(obj)


@canonical.register
def _(obj: str) -> Json:
    return str.__str__(obj)


@canonical.register
def _(obj: float) -> Json:
    if not math.isfinite(obj):
        raise ValueError(f"Non-finite float cannot be digested: {obj!r}")
    return float(obj)


@canonical.register
def _(obj: Enum) -> Json:
    return canonical(obj.value)


@canonical.register
def _(obj: PurePath) -> Json:
    raise TypeError(
        f"unmarked Path {obj!s}: annotate the field with InputData, InputConfig or Cosmetic"
    )


@canonical.register
def _(obj: Mapping) -> Json:
    out: dict[str, Json] = {}
    for key, value in obj.items():
        text = canonical(key)
        if not isinstance(text, str):
            raise TypeError(f"Mapping keys must canonicalize to str, got {key!r}")
        out[text] = canonical(value)
    return out


@canonical.register(list)
@canonical.register(tuple)
def _(obj: list | tuple) -> Json:
    return [canonical(item) for item in obj]


@canonical.register(set)
@canonical.register(frozenset)
def _(obj: set | frozenset) -> Json:
    return sorted((canonical(item) for item in obj), key=_dumps)


@canonical.register
def _(obj: BaseModel) -> Json:
    cls = type(obj)
    names = list(cls.model_fields)
    markers = {n: tuple(cls.model_fields[n].metadata) for n in names}
    fields = _fields(obj, names, markers)
    assert isinstance(fields, dict)
    if obj.model_extra:
        fields.update({str(k): canonical(v) for k, v in obj.model_extra.items()})
    return fields


def _annotated_markers(annotation: object) -> tuple[object, ...]:
    """Return the metadata of an ``Annotated[...]`` annotation (else empty)."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[1:]
    return ()


def _fields(obj: object, names: list[str], metadata: Mapping[str, tuple[object, ...]]) -> Json:
    """Canonicalize the named fields of ``obj`` honoring their markers."""
    out: dict[str, Json] = {}
    for name in names:
        marker = _select_marker(metadata[name])
        if isinstance(marker, Cosmetic):
            continue
        out[name] = _field_value(getattr(obj, name), marker)
    return out


def _select_marker(metadata: tuple[object, ...]) -> Marker | None:
    found = [m for m in metadata if isinstance(m, Cosmetic | InputData | InputConfig)]
    if len(found) > 1:
        raise TypeError(f"Field carries multiple identity markers: {found!r}")
    return found[0] if found else None


def _field_value(value: object, marker: Marker | None) -> Json:
    match marker:
        case None:
            return canonical(value)
        case _ if value is None:
            return None
        case InputData():
            return content_digest(_as_path(value))
        case InputConfig():
            return toml_digest(_as_path(value))
        case _:
            raise TypeError(f"Unsupported marker {marker!r}")


def _as_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise TypeError(f"Input marker requires a Path, got {type(value).__name__}")


def content_digest(path: Path) -> Digest:
    """Digest a file's bytes, or a directory's recursive content.

    A directory digest covers sorted ``(relative posix path, file digest)``
    pairs, so it does not depend on where the directory lives.

    Args:
        path: File or directory.

    Returns:
        ``"sha256:<hex>"`` digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if path.is_dir():
        entries = sorted(p for p in path.rglob("*") if p.is_file())
        pairs = [[p.relative_to(path).as_posix(), _file_digest(p)] for p in entries]
        return canonical_digest(pairs)
    if path.is_file():
        return _file_digest(path)
    raise FileNotFoundError(path)


def _file_digest(path: Path) -> Digest:
    with path.open("rb") as fh:
        return _format(hashlib.file_digest(fh, _ALGORITHM).hexdigest())


def toml_digest(path: Path) -> Digest:
    """Semantic digest of a TOML file.

    The file is parsed and canonicalized, so comments, whitespace and key order
    do not matter. String values naming an existing ``*.toml`` file (relative to
    the referencing file's directory) are replaced by that file's parsed
    content, recursively.

    Args:
        path: TOML file.

    Returns:
        ``"sha256:<hex>"`` digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If TOML references form a cycle.
    """
    return canonical_digest(_toml_structure(path, frozenset()))


def _toml_structure(path: Path, chain: frozenset[Path]) -> Json:
    resolved = path.resolve()
    if resolved in chain:
        raise ValueError(f"Cyclic TOML reference at {resolved}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return _inline_references(data, resolved.parent, chain | {resolved})


def _inline_references(value: Any, base: Path, chain: frozenset[Path]) -> Json:
    match value:
        case dict():
            return {k: _inline_references(v, base, chain) for k, v in value.items()}
        case list():
            return [_inline_references(v, base, chain) for v in value]
        case str() if (ref := _toml_reference(value, base)) is not None:
            return _toml_structure(ref, chain)
        case _ if hasattr(value, "isoformat"):
            return value.isoformat()
        case _:
            return canonical(value)


def _toml_reference(text: str, base: Path) -> Path | None:
    if not text.endswith(".toml"):
        return None
    try:
        candidate = base / text
        return candidate if candidate.is_file() else None
    except OSError, ValueError:
        return None
