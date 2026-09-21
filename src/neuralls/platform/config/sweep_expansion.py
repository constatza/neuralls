"""Expand ``*.sweep.toml`` specs into real per-value dataset configs.

A sweep spec is an ordinary dataset config except exactly one field inside its
single ``[[generation.strategy]]`` table is a list. Each value is rendered out
as its own dataset TOML inside a gitignored ``_generated/`` directory sibling
to the spec. ``ensure_generated_dataset_config`` materializes those files on
demand when a case config references one that does not exist yet; the
``scripts/expand_dataset_sweep.py`` CLI exposes the same functions manually.
See that script's docstring for the sweep spec format.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from neuralls.shared.constants import SWEEP_GENERATED_SUBDIR_NAME

_SWEEP_SUFFIX = ".sweep.toml"

# Name of the gitignored subdirectory expanded dataset configs are written
# into by default, sibling to their `*.sweep.toml` source. Generated files
# are build artifacts (100% mechanically derivable from their sweep spec),
# never tracked in git. Kept in sync with the `configs/datasets/*/*/_generated/`
# pattern in .gitignore — if this default changes, update that pattern too.
# Also shared with error-message hints in loaders.py/assembler.py.
_GENERATED_SUBDIR_NAME = SWEEP_GENERATED_SUBDIR_NAME

TomlValue = bool | int | float | str | list[Any]


def _toml_scalar(value: TomlValue) -> str:
    """Render one Python value as a TOML literal (str/bool/int/float/list only)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise ValueError(f"unsupported value type in sweep spec: {type(value).__name__}")


def _render_table(
    table: dict[str, TomlValue], *, comments: dict[str, list[str]] | None = None
) -> str:
    """Render a flat dict as TOML ``key = value`` lines, preserving key order.

    Args:
        table: The fields to render.
        comments: Optional per-key comment lines, rendered as `# `-prefixed
            lines immediately before that key's line — e.g. the source-level
            note above `exclude_indices` in a parametric-family dataset.
    """
    comments = comments or {}
    lines: list[str] = []
    for key, value in table.items():
        lines.extend(f"# {line}" for line in comments.get(key, []))
        lines.append(f"{key} = {_toml_scalar(value)}")
    return "\n".join(lines)


def _find_swept_field(strategy: dict[str, TomlValue]) -> tuple[str, list[Any]]:
    """Return the (field name, values) of the one list-valued strategy field."""
    list_fields = [(key, value) for key, value in strategy.items() if isinstance(value, list)]
    if not list_fields:
        raise ValueError(
            "no list-valued field found in [[generation.strategy]] — expected exactly "
            "one field (e.g. samples or cg_iters) to be a list"
        )
    if len(list_fields) > 1:
        names = ", ".join(key for key, _ in list_fields)
        raise ValueError(f"only one field may be swept at a time, found lists for: {names}")
    axis, values = list_fields[0]
    return axis, values


def _render_dataset_toml(
    *,
    id_: str,
    source: dict[str, TomlValue],
    generation_header: dict[str, TomlValue],
    strategy: dict[str, TomlValue],
    strategy_comment: list[str] | None,
    output: dict[str, TomlValue],
    source_comments: dict[str, list[str]] | None = None,
) -> str:
    """Render one expanded dataset config, matching the hand-written file shape exactly."""
    parts = [
        f"id = {_toml_scalar(id_)}",
        "",
        "[source]",
        _render_table(source, comments=source_comments),
        "",
        "[generation]",
        _render_table(generation_header),
        "",
    ]
    if strategy_comment:
        parts.extend(f"# {line}" for line in strategy_comment)
    parts.extend(
        [
            "[[generation.strategy]]",
            _render_table(strategy),
            "",
            "[output]",
            _render_table(output),
        ]
    )
    return "\n".join(parts) + "\n"


def expand_sweep(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    filename_stem: str,
) -> list[Path]:
    """Expand one parsed sweep spec into real per-value dataset TOML files."""
    base_id = spec["id"]
    source = spec["source"]
    generation = spec["generation"]
    output = spec["output"]
    strategy_comment = spec.get("strategy_comment")
    source_comments = spec.get("source_comments")

    strategies = generation.get("strategy", [])
    if len(strategies) != 1:
        raise ValueError(
            f"sweep spec must declare exactly one [[generation.strategy]] entry, "
            f"found {len(strategies)}"
        )
    strategy = strategies[0]
    axis, values = _find_swept_field(strategy)
    if not values:
        raise ValueError(f"swept field {axis!r} must be a non-empty list")

    id_template = spec.get("id_template", f"{base_id}-{{value}}")
    filename_template = spec.get("filename_template", f"{filename_stem}-{{value}}.toml")

    generation_header = {key: value for key, value in generation.items() if key != "strategy"}
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for value in values:
        content = _render_dataset_toml(
            id_=id_template.format(value=value),
            source=source,
            generation_header=generation_header,
            strategy={**strategy, axis: value},
            strategy_comment=strategy_comment,
            output=output,
            source_comments=source_comments,
        )
        path = output_dir / filename_template.format(value=value)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def _load_sweep_spec(path: Path) -> dict[str, Any]:
    """Load and lightly validate a ``*.sweep.toml`` file."""
    if not path.name.endswith(_SWEEP_SUFFIX):
        raise ValueError(f"expected a '{_SWEEP_SUFFIX}' file, got: {path.name}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _default_output_dir(sweep_file: Path, generated_subdir_name: str) -> Path:
    """Return the default expansion output directory for one sweep file.

    Args:
        sweep_file: Path to a ``*.sweep.toml`` file.
        generated_subdir_name: Name of the gitignored subdirectory generated
            dataset configs are written into, sibling to ``sweep_file``.

    Returns:
        ``sweep_file``'s parent directory joined with ``generated_subdir_name``.
    """
    return sweep_file.parent / generated_subdir_name


def expand_all(root: Path, *, generated_subdir_name: str = _GENERATED_SUBDIR_NAME) -> list[Path]:
    """Expand every ``*.sweep.toml`` file found recursively under ``root``.

    Each sweep file's outputs are written into its own sweep-file-relative
    default output directory (see :func:`_default_output_dir`) — mirroring
    running the single-file CLI mode once per discovered sweep file.

    Args:
        root: Directory to search recursively for ``*.sweep.toml`` files.
        generated_subdir_name: Name of the gitignored subdirectory each sweep
            file's outputs are written into, sibling to that file.

    Returns:
        Paths of every dataset config file written, across all sweep files.
    """
    written: list[Path] = []
    for sweep_file in sorted(root.rglob(f"*{_SWEEP_SUFFIX}")):
        spec = _load_sweep_spec(sweep_file)
        output_dir = _default_output_dir(sweep_file, generated_subdir_name)
        filename_stem = sweep_file.name[: -len(_SWEEP_SUFFIX)]
        written.extend(expand_sweep(spec, output_dir=output_dir, filename_stem=filename_stem))
    return written


def ensure_generated_dataset_config(path: Path, *, force: bool = False) -> None:
    """Materialize a sweep-generated dataset config from its sweep source.

    A no-op unless ``path`` lives under a ``_generated/`` directory. Without
    ``force`` it also stays a no-op while ``path`` exists; with ``force`` the
    sibling sweeps are always re-expanded, overwriting existing outputs.
    Every ``*.sweep.toml`` next to the ``_generated/`` directory is expanded; if
    none produces ``path`` the caller's own not-found error stands.

    Args:
        path: Dataset config path a case config references.
        force: Recreate the generated files even when ``path`` already exists.
    """
    if _GENERATED_SUBDIR_NAME not in path.parts or (path.exists() and not force):
        return
    generated_dir = next(p for p in (path, *path.parents) if p.name == _GENERATED_SUBDIR_NAME)
    for sweep_file in sorted(generated_dir.parent.glob(f"*{_SWEEP_SUFFIX}")):
        expand_sweep(
            _load_sweep_spec(sweep_file),
            output_dir=generated_dir,
            filename_stem=sweep_file.name[: -len(_SWEEP_SUFFIX)],
        )
