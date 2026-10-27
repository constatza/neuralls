"""ConfigContext and path-expansion helpers used by all Pydantic config validators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationInfo

from neuralls.platform.config.resolution import expand_user_path, resolve_local_path
from neuralls.platform.config.settings import NeurallsSettings

_GRAPH_CG_RE = re.compile(r"\$\{GRAPH_CG_[^}]+\}")
_NEURALLS_RE = re.compile(r"\$\{NEURALLS_([^}]+)\}")
_GLOB_CHARS = frozenset("*?[]")


@dataclass(frozen=True, slots=True)
class ConfigContext:
    """Immutable context passed to Pydantic model validators via model_validate(context=...).

    Carries the path of the config file being loaded and the resolved runtime settings
    so validators can expand ${NEURALLS_*} placeholders and anchor relative paths.

    Attributes:
        config_path: Absolute path to the TOML file being loaded.
        settings: Resolved NeurallsSettings for this invocation.
    """

    config_path: Path
    settings: NeurallsSettings

    @property
    def config_dir(self) -> Path:
        """Parent directory of config_path — used to anchor relative paths."""
        return self.config_path.parent

    def as_pydantic_context(self) -> dict[str, object]:
        """Serialize to the dict format expected by Pydantic v2 model_validate(context=...)."""
        return {
            "config_path": self.config_path,
            "config_dir": self.config_dir,
            "settings": self.settings,
        }

    @staticmethod
    def from_pydantic_context(ctx: dict[str, object]) -> ConfigContext:
        """Reconstruct from the dict produced by as_pydantic_context()."""
        return ConfigContext(
            config_path=cast(Path, ctx["config_path"]),
            settings=cast(NeurallsSettings, ctx["settings"]),
        )


def expand_config_path(value: str, ctx: ConfigContext) -> str:
    """Expand ${NEURALLS_*} placeholders and anchor relative paths to config dir."""
    if m := _GRAPH_CG_RE.search(value):
        raise ValueError(
            f"Stale placeholder '{m.group()}' in config. "
            "Rename GRAPH_CG_* → NEURALLS_* (see docs/env-reference.md)."
        )

    def _sub(m: re.Match[str]) -> str:
        attr = m.group(1).lower()
        val = getattr(ctx.settings, attr, None)
        if val is None:
            raise ValueError(
                f"Unknown placeholder '${{NEURALLS_{m.group(1)}}}'. "
                "Available: RAW_DIR, PROCESSED_DIR, OUTPUT_DIR."
            )
        return str(val)

    expanded = str(expand_user_path(_NEURALLS_RE.sub(_sub, value)))
    return str(resolve_local_path(expanded, base_dir=ctx.config_dir))


def expand_path_field(v: object, info: ValidationInfo, *, glob: bool = False) -> object:
    """Shared body for `@field_validator(..., mode="before")` path/glob expansion.

    Args:
        v: Raw field value from Pydantic validation.
        info: Pydantic validation context, carrying the config resolution context.
        glob: If True, expand as a glob pattern via `expand_config_glob`; otherwise
            expand as a plain path via `expand_config_path`.

    Returns:
        The expanded string, or `v` unchanged if it isn't a string or no context is set.
    """
    if v is None or info.context is None or not isinstance(v, str):
        return v
    ctx = ConfigContext.from_pydantic_context(info.context)
    return expand_config_glob(v, ctx) if glob else expand_config_path(v, ctx)


def _expand_user_path(value: str) -> str:
    """Compatibility wrapper for shared HOME-first user-path expansion."""
    return str(expand_user_path(value))


def _join_glob_suffix(prefix: str, suffix: str) -> str:
    """Join a resolved path prefix with a glob suffix without reparsing Windows paths."""
    cleaned_prefix = prefix.rstrip("/\\")
    cleaned_suffix = suffix.lstrip("/\\")
    if not cleaned_prefix:
        return cleaned_suffix
    separator = "\\" if "\\" in cleaned_prefix and "/" not in cleaned_prefix else "/"
    return f"{cleaned_prefix}{separator}{cleaned_suffix}"


def expand_config_glob(value: str, ctx: ConfigContext) -> str:
    """Like expand_config_path but preserves glob wildcards in the filename."""
    for i, ch in enumerate(value):
        if ch in _GLOB_CHARS:
            prefix = value[:i]
            suffix = value[i:]
            if not prefix:
                return _join_glob_suffix(str(ctx.config_dir), suffix)
            resolved_prefix = (
                expand_config_path(prefix, ctx) if prefix.strip() else str(ctx.config_dir)
            )
            if prefix.endswith(("/", "\\")):
                return _join_glob_suffix(resolved_prefix, suffix)
            return f"{resolved_prefix}{suffix}"
    return expand_config_path(value, ctx)
