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
"0-CG" case in `configs/datasets/train/*/gaussian-0cg*.toml`).

Example:
    uv run python scripts/expand_dataset_sweep.py \\
        configs/datasets/train/rectangular-high-condition/gaussian-cg50.sweep.toml
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

_SWEEP_SUFFIX = ".sweep.toml"

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


def _render_table(table: dict[str, TomlValue]) -> str:
    """Render a flat dict as TOML ``key = value`` lines, preserving key order."""
    return "\n".join(f"{key} = {_toml_scalar(value)}" for key, value in table.items())


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
) -> str:
    """Render one expanded dataset config, matching the hand-written file shape exactly."""
    parts = [
        f"id = {_toml_scalar(id_)}",
        "",
        "[source]",
        _render_table(source),
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

    generation_header = {key: value for key, value in generation.items() if key != "strategy"}
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for value in values:
        content = _render_dataset_toml(
            id_=f"{base_id}-{value}",
            source=source,
            generation_header=generation_header,
            strategy={**strategy, axis: value},
            strategy_comment=strategy_comment,
            output=output,
        )
        path = output_dir / f"{filename_stem}-{value}.toml"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def _load_sweep_spec(path: Path) -> dict[str, Any]:
    """Load and lightly validate a ``*.sweep.toml`` file."""
    if not path.name.endswith(_SWEEP_SUFFIX):
        raise ValueError(f"expected a '{_SWEEP_SUFFIX}' file, got: {path.name}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_file", type=Path, help="Path to a *.sweep.toml file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write expanded dataset configs into (default: sweep_file's own directory).",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    sweep_file: Path = args.sweep_file
    output_dir: Path = args.output_dir or sweep_file.parent
    filename_stem = sweep_file.name[: -len(_SWEEP_SUFFIX)]

    try:
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
