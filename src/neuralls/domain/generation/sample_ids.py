"""Sample-id derivation and enumeration rules for glob-matched source files."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

_GLOB_CHARS = ("*", "?", "[", "]")
_DEFAULT_SAMPLE_ID_REGEX = r"(\d+)(?!.*\d)"


class EnumerateBy(StrEnum):
    """Criterion for assigning sequential IDs to glob-matched files.

    Use when filenames carry no natural integer ID (e.g. parameter-encoded names
    like ``E1_3000_E2_78000_matrix.txt``).  Files are sorted by the chosen
    criterion and assigned sequential IDs 0, 1, 2, …
    """

    NAME = "name"
    CTIME = "ctime"
    MTIME = "mtime"


def _sort_key_for(path: Path, by: EnumerateBy) -> float | str:
    """Return the sort key for *path* under the given *by* strategy."""
    match by:
        case EnumerateBy.NAME:
            return path.name
        case EnumerateBy.CTIME:
            return path.stat().st_birthtime
        case EnumerateBy.MTIME:
            return path.stat().st_mtime


def _enumerate_files(paths: Sequence[Path], by: EnumerateBy) -> dict[int, Path]:
    """Assign sequential IDs to *paths* sorted by *by*, returning ``{id: path}``."""
    return {i: p for i, p in enumerate(sorted(paths, key=lambda p: _sort_key_for(p, by)))}


def _filter_mapping(
    mapping: dict[int, Path],
    *,
    include_indices: tuple[int, ...] | None,
    exclude_indices: tuple[int, ...],
) -> dict[int, Path]:
    """Restrict *mapping* to `include_indices`, or drop `exclude_indices`.

    Keeps original sample ids as dict keys (no renumbering) so downstream
    keyed lookups (`bind_sources`, `load_dense_sample`) stay correct.

    # ponytail: hand-maintained id lists per dataset TOML are a crude,
    # manual train/eval split — refine into a shared, seeded split utility
    # (e.g. fractional or stratified) if more parametric-family cases need
    # this, so the held-out ids aren't hardcoded and duplicated per config.
    """
    if include_indices is not None and exclude_indices:
        raise ValueError("include_indices and exclude_indices are mutually exclusive.")
    if include_indices is not None:
        missing = set(include_indices) - mapping.keys()
        if missing:
            raise ValueError(f"include_indices references unknown sample ids: {sorted(missing)}")
        return {i: mapping[i] for i in include_indices}
    if exclude_indices:
        missing = set(exclude_indices) - mapping.keys()
        if missing:
            raise ValueError(f"exclude_indices references unknown sample ids: {sorted(missing)}")
        return {i: p for i, p in mapping.items() if i not in exclude_indices}
    return mapping


def _is_glob_expression(expr: str) -> bool:
    """Return True when the path expression contains glob meta characters."""
    return any(char in expr for char in _GLOB_CHARS)


def _split_glob_root(expr: str) -> tuple[Path, str]:
    """Split a glob expression into its literal root dir and glob pattern.

    Everything up to the first path segment containing a glob character is
    the literal root (must already exist); everything from there on,
    rejoined with "/", is handed to ``Path.glob()`` as-is. ``Path.glob``
    already matches multi-segment and recursive ("**") patterns on its own,
    so no custom traversal is needed once the split point is right.
    """
    parts = Path(expr).parts
    for i, part in enumerate(parts):
        if _is_glob_expression(part):
            return Path(*parts[:i]), "/".join(parts[i:])
    return Path(expr).parent, Path(expr).name


def _id_source_text(path: Path, *, leaf_is_literal: bool) -> str:
    """Text ``sample_id_regex`` searches for the sample id.

    When the glob's leaf (filename) segment is a literal constant (the
    wildcard lives in a parent directory instead, e.g. ``*/K_ff.mtx``), the
    id is the matched directory's name. Otherwise (today's filename-glob
    case, e.g. ``A_*.txt``) it is the matched file's stem, as before.
    """
    return path.parent.name if leaf_is_literal else path.stem


def _extract_sample_id(path: Path, pattern: re.Pattern[str], *, leaf_is_literal: bool) -> int:
    """Extract integer sample id from wherever the glob's wildcard put it."""
    text = _id_source_text(path, leaf_is_literal=leaf_is_literal)
    match = pattern.search(text)
    if match is None:
        raise ValueError(
            f"Could not extract sample id from '{text}' (derived from {path}) "
            f"using regex '{pattern.pattern}'."
        )
    return int(match.group(1))


def _index_by_sample_id_regex(
    paths: Sequence[Path],
    *,
    noun: str,
    sample_id_regex: str | None,
    leaf_is_literal: bool,
) -> dict[int, Path]:
    """Map glob-derived sample ids to *paths*, rejecting duplicate ids."""
    regex = re.compile(sample_id_regex or _DEFAULT_SAMPLE_ID_REGEX)
    mapping: dict[int, Path] = {}
    for path in paths:
        sample_id = _extract_sample_id(path, regex, leaf_is_literal=leaf_is_literal)
        if sample_id in mapping:
            raise ValueError(
                f"Duplicate {noun} sample id {sample_id} for files {mapping[sample_id]} and {path}"
            )
        mapping[sample_id] = path
    return mapping


def _build_glob_index(
    expr: str,
    *,
    noun: str,
    sample_id_regex: str | None,
    enumerate_by: EnumerateBy | None,
    include_indices: tuple[int, ...] | None,
    exclude_indices: tuple[int, ...],
) -> dict[int, Path]:
    """Resolve a glob expression to ``{sample_id: path}``.

    Args:
        expr: Glob expression. Everything before its first wildcard segment
            is the literal root directory, which must already exist; the
            rest may span multiple path segments and use "**" for recursive
            matching (see ``_split_glob_root``).
        noun: Source kind used in error messages ("matrix" or "vector").
        sample_id_regex: Regex whose first group holds the id. Searched
            against the matched directory's name when the glob's leaf
            (filename) segment is a literal constant, otherwise against the
            matched file's stem (see ``_id_source_text``). Ignored when
            *enumerate_by* is given; defaults to the trailing integer.
        enumerate_by: Assign sequential ids by this criterion instead of parsing
            them out of the filenames.
        include_indices: Keep only these sample ids, if given.
        exclude_indices: Drop these sample ids.

    Returns:
        Mapping of sample id to source file, without renumbering.
    """
    root, pattern = _split_glob_root(expr)
    if not root.exists():
        raise FileNotFoundError(f"{noun.capitalize()} glob root directory not found: {root}")
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {noun} files match glob: {expr}")
    mapping = (
        _enumerate_files(paths, enumerate_by)
        if enumerate_by is not None
        else _index_by_sample_id_regex(
            paths,
            noun=noun,
            sample_id_regex=sample_id_regex,
            leaf_is_literal=not _is_glob_expression(pattern.rsplit("/", 1)[-1]),
        )
    )
    mapping = _filter_mapping(
        mapping, include_indices=include_indices, exclude_indices=exclude_indices
    )
    if not mapping:
        raise ValueError(f"No {noun} samples remain after filtering glob: {expr}")
    return mapping
