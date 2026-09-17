r"""Expand a list-valued dataset config into real per-value dataset TOMLs.

`configs/README.md` requires one real dataset config per real dataset
variant (no "family" abstraction the runtime resolves at load time — see
`platform/config/models/data_models.py:DataConfigFile.id`, a required field
read from a literal file on disk by every consumer). A samples/cg_iters
sweep across N values still needs N real files; this script only removes
the need to hand-type them.

A sweep spec (`*.sweep.toml`) is an ordinary dataset config except exactly
one field inside its single `[[generation.strategy]]` table is a list
instead of a scalar (e.g. `samples = [1000, 2000, 5000, 10000]`). Every
other field — `[source]`, `[generation]`'s scalar fields, `[output]`, and
every other strategy field — is rendered back out verbatim: this script has
no knowledge of "samples", "cg_iters", "gaussian_residuals",
"gaussian_forward", or any other strategy/field name. It only knows "one
list-valued field in the strategy table, everything else passed through."
An optional top-level `strategy_comment` (a list of lines) is reproduced
verbatim as `#`-comments directly above `[[generation.strategy]]`, for
sweeps whose strategy choice needs an explanation (see the `gaussian_forward`
"0-CG" case in `configs/datasets/train/*/gaussian-cg0*.toml`). An optional
top-level `[source_comments]` table maps a `[source]` field name to comment
lines rendered directly above that field (see `exclude_indices` in
`configs/datasets/train/45x15randomE/gaussian-cg.sweep.toml`).

By default each expanded file's id/filename is `{id}-{value}`/`{stem}-{value}.toml`
— right for a trailing axis like `samples` (`gaussian-cg50-1000`). A swept
axis that belongs in the *middle* of the id instead (e.g. `stop`, where the
convention is `gaussian-cg10-45x15`, not `gaussian-cg-45x15-10`) can override
this with optional top-level `id_template`/`filename_template` fields
containing a `{value}` placeholder, e.g. `id_template = "gaussian-cg{value}-45x15"`,
`filename_template = "gaussian-cg{value}.toml"` — the same `{value}`-placeholder
convention `[[dataset_sweeps]].path_template` already uses in case configs.

By default, expanded files are written next to their sweep spec, inside a
gitignored ``_generated/`` subdirectory (see ``_GENERATED_SUBDIR_NAME``) —
generated dataset configs are 100% mechanically derivable from their
``*.sweep.toml`` source, so they are build artifacts, not tracked content.
Pass ``--output-dir`` to override this for a single-file run, or
``--generated-subdir-name`` to rename the convention itself.

``--all [ROOT]`` expands every ``*.sweep.toml`` file found recursively under
``ROOT`` (default: ``configs``) in one invocation, each into its own
sweep-file-relative ``_generated/`` subdirectory — this is the normal way to
regenerate everything after a fresh checkout or a source-file edit. It is
mutually exclusive with the single-file positional argument.

Example:
    uv run python scripts/expand_dataset_sweep.py \\
        configs/datasets/train/rectangular-high-condition/gaussian-cg50.sweep.toml

    uv run python scripts/expand_dataset_sweep.py --all
"""

from __future__ import annotations

import argparse
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a single *.sweep.toml file. Mutually exclusive with --all.",
    )
    parser.add_argument(
        "--all",
        dest="all_root",
        type=Path,
        nargs="?",
        const=Path("configs"),
        default=None,
        metavar="ROOT",
        help=(
            "Expand every *.sweep.toml file found recursively under ROOT "
            "(default: 'configs') instead of a single sweep_file. Mutually "
            "exclusive with sweep_file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write expanded dataset configs into. Single-file mode only "
            f"(default: sweep_file's own directory / '{_GENERATED_SUBDIR_NAME}')."
        ),
    )
    parser.add_argument(
        "--generated-subdir-name",
        type=str,
        default=_GENERATED_SUBDIR_NAME,
        help=(
            "Name of the gitignored subdirectory generated dataset configs are written "
            f"into by default, in either mode (default: {_GENERATED_SUBDIR_NAME!r})."
        ),
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.sweep_file is not None and args.all_root is not None:
        parser.error("sweep_file and --all are mutually exclusive")
    if args.sweep_file is None and args.all_root is None:
        parser.error("one of sweep_file or --all is required")
    if args.all_root is not None and args.output_dir is not None:
        parser.error("--output-dir is not supported together with --all")

    try:
        if args.all_root is not None:
            written = expand_all(args.all_root, generated_subdir_name=args.generated_subdir_name)
        else:
            sweep_file: Path = args.sweep_file
            output_dir = args.output_dir or _default_output_dir(
                sweep_file, args.generated_subdir_name
            )
            filename_stem = sweep_file.name[: -len(_SWEEP_SUFFIX)]
            spec = _load_sweep_spec(sweep_file)
            written = expand_sweep(spec, output_dir=output_dir, filename_stem=filename_stem)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
