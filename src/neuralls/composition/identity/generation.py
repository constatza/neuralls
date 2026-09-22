"""Generation-stage identity: the dataset config plus the content of its raw sources."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from neuralls.domain.generation.source_streams import _is_glob_expression
from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.shared.digest import Digest, canonical_digest, content_digest

_STRATEGY_PATH_KEYS = ("rhs_glob", "rhs_path", "solutions_glob", "solutions_path")


def _declared_expressions(cfg: DataConfigFile) -> Iterator[tuple[str, str]]:
    """Yield ``(stable label, path expression)`` for every input file the config declares.

    A superset of what the archive resolvers pick, so a source is never
    missed; the labels are config-structural, never absolute paths.
    """
    source = cfg.source
    for field in ("matrix_path", "rhs_path", "solutions_path", "solution_path"):
        if (value := getattr(source, field)) is not None:
            yield f"source.{field}", value
    for index, expr in enumerate(source.parameters_paths):
        yield f"source.parameters_paths[{index}]", expr
    if cfg.generation.rhs_archive_glob is not None:
        yield "generation.rhs_archive_glob", cfg.generation.rhs_archive_glob
    for index, strategy in enumerate(cfg.generation.strategy):
        options = strategy.model_dump(exclude_none=True)
        for key in _STRATEGY_PATH_KEYS:
            if isinstance(options.get(key), str):
                yield f"generation.strategy[{index}].{key}", options[key]


def _expand(expr: str) -> list[Path]:
    """Resolve a path expression to existing files, failing loudly when none exist."""
    path = Path(expr)
    if not _is_glob_expression(expr):
        if not path.exists():
            raise FileNotFoundError(f"Declared generation source not found: {path}")
        return [path]
    if not path.parent.exists():
        raise FileNotFoundError(f"Declared generation source directory not found: {path.parent}")
    matches = sorted(path.parent.glob(path.name))
    if not matches:
        raise FileNotFoundError(f"No generation source files match glob: {expr}")
    return matches


def _expression_digest(expr: str) -> Digest:
    """Digest an expression's files by content.

    Glob matches keep their file names (they encode sample ids); a single
    file contributes its suffix (it selects the parser) but not its location.
    """
    files = _expand(expr)
    if not _is_glob_expression(expr):
        return canonical_digest(Path(expr).suffix, content_digest(files[0]))
    return canonical_digest([[file.name, content_digest(file)] for file in files])


def sources_digest(cfg: DataConfigFile) -> Digest:
    """Digest the content of every raw source file a data config declares.

    Args:
        cfg (DataConfigFile): Resolved data config.

    Returns:
        Digest: Digest over ``{label: content digest}``.

    Raises:
        FileNotFoundError: If a declared source does not exist.
    """
    return canonical_digest(
        {label: _expression_digest(expr) for label, expr in _declared_expressions(cfg)}
    )


def generation_identity(data_cfg: DataConfigFile) -> StageIdentity:
    """Derive the identity of the dataset a data config would generate.

    Cosmetic config fields (output directory, id) do not affect the key;
    everything else in the config and the bytes of every raw source do.

    Args:
        data_cfg (DataConfigFile): Resolved data config.

    Returns:
        StageIdentity: ``generation`` identity with ``config`` and ``sources`` components.

    Raises:
        FileNotFoundError: If a declared source does not exist.
    """
    return StageIdentity.build(
        "generation",
        {"config": canonical_digest(data_cfg), "sources": sources_digest(data_cfg)},
    )
