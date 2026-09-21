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
from pathlib import Path

from neuralls.platform.config.sweep_expansion import (  # noqa: F401  (re-exported for tests)
    _GENERATED_SUBDIR_NAME,
    _SWEEP_SUFFIX,
    _default_output_dir,
    _find_swept_field,
    _load_sweep_spec,
    _render_dataset_toml,
    _toml_scalar,
    expand_all,
    expand_sweep,
)


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
