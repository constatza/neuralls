"""ConfigContext and path-anchoring helpers used by all Pydantic config validators."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

from pydantic import ValidationInfo

from neuralls.platform.config.resolution import (
    is_windows_absolute_path,
    normalize_windows_absolute_path,
)
from neuralls.platform.config.settings import NeurallsSettings

_GRAPH_CG_RE = re.compile(r"\$\{GRAPH_CG_[^}]+\}")
_NEURALLS_RE = re.compile(r"\$\{NEURALLS_([^}]+)\}")
_GLOB_CHARS = frozenset("*?[]")
# Two or more scheme characters, so a Windows drive letter (`C:\`) is never a scheme.
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")
# `C:x` and `a:b/c` name a drive's working directory, which a pure anchor cannot know.
_DRIVE_RELATIVE_RE = re.compile(r"^[A-Za-z]:(?![\\/])")
_LOCAL_FILE_HOSTS = frozenset({"", "localhost"})
_HOME_PREFIXES = ("~/", "~\\") if os.name == "nt" else ("~/",)


@dataclass(frozen=True, slots=True)
class ConfigContext:
    """Immutable context passed to Pydantic model validators via model_validate(context=...).

    Carries the path of the config file being loaded, the user's home directory and the
    resolved runtime settings so validators can expand ${NEURALLS_*} placeholders, expand
    a leading ``~`` and anchor relative paths without touching the filesystem or the process
    environment. The loaders (the disk edge) fill ``config_path`` and ``home_dir`` with
    absolute paths; a test passes relative ones and gets relative results.

    A validator anchors its path whenever a context is given. A model built in memory
    without a context is taken as already anchored and keeps its paths as written.

    Attributes:
        config_path: Path to the TOML file being loaded (absolute when a loader built it).
        settings: Resolved NeurallsSettings for this invocation.
        home_dir: Directory a leading ``~`` in a config path stands for.
    """

    config_path: Path
    settings: NeurallsSettings
    home_dir: Path

    @classmethod
    def for_file(cls, path: Path, settings: NeurallsSettings) -> ConfigContext:
        """Context for loading the config file at ``path``: the disk edge.

        Reads the filesystem and the process environment, which `anchor` never does:
        ``path.resolve()`` makes the config path absolute and ``Path.home()`` names the
        home directory a leading ``~`` stands for.
        """
        return cls(config_path=path.resolve(), settings=settings, home_dir=Path.home())

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
            "home_dir": self.home_dir,
        }

    @staticmethod
    def from_pydantic_context(ctx: dict[str, object]) -> ConfigContext:
        """Reconstruct from the dict produced by as_pydantic_context()."""
        return ConfigContext(
            config_path=cast(Path, ctx["config_path"]),
            settings=cast(NeurallsSettings, ctx["settings"]),
            home_dir=cast(Path, ctx["home_dir"]),
        )


def substitute_placeholders(value: str, ctx: ConfigContext) -> str:
    """Replace ${NEURALLS_*} placeholders from settings; reject stale ${GRAPH_CG_*} ones."""
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

    return _NEURALLS_RE.sub(_sub, value)


def _strip_leading_dot_segments(path: str) -> str:
    """Drop leading ``./`` and ``../`` so a URI path cannot climb out of its anchor."""
    while path.startswith("./"):
        path = path[2:]
    while path.startswith("../"):
        path = path[3:]
    return path


def _uri_path(uri: str) -> str:
    """Extract the local path a ``file:`` or ``sqlite:`` URI names, without touching disk.

    Mirrors dlkit's ``url_resolver``: ``file:///abs`` and ``sqlite:////abs`` are absolute,
    ``sqlite:///rel`` is relative to the anchor, and ``file://`` with a host is rejected.
    """
    parts = urlsplit(uri)
    scheme = parts.scheme.lower()
    match scheme:
        case "file":
            if parts.netloc not in _LOCAL_FILE_HOSTS:
                raise ValueError(
                    f"File URL has non-empty host '{parts.netloc}': '{uri}'. "
                    "Use file:///path (3 slashes) for a local file."
                )
            path = unquote(parts.path)
            if path.startswith("//") and not path.startswith("///"):
                return f"/{path.lstrip('/')}"
            if path.startswith("/"):
                return path
            return _strip_leading_dot_segments(path)
        case "sqlite":
            path = unquote(parts.path)
            if path.startswith("//") and not path.startswith("///"):
                return path[1:]
            if path.startswith("/"):
                return _strip_leading_dot_segments(path[1:])
            return _strip_leading_dot_segments(path)
        case _:
            raise ValueError(f"Unsupported scheme for local resolution: {scheme}")


def _expand_home(value: str, ctx: ConfigContext) -> Path | None:
    """Expand a leading ``~`` from ``ctx.home_dir``; None when ``value`` has none."""
    if value == "~":
        return ctx.home_dir
    if value.startswith(_HOME_PREFIXES):
        return ctx.home_dir / value[2:]
    if value.startswith("~"):
        raise ValueError(
            f"Cannot expand '{value}': only '~' and '~/...' are supported, not '~user'."
        )
    return None


def _reject_drive_relative(text: str) -> None:
    """Reject `C:x`: it names the working directory of a drive, which a pure anchor cannot know."""
    if _DRIVE_RELATIVE_RE.match(text):
        raise ValueError(
            f"Drive-relative path '{text}' is not supported: it depends on the current "
            "directory of that drive. Use 'C:/dir/file' or a path relative to the config file."
        )


def _reject_posix_absolute_on_windows(text: str) -> None:
    """Reject a drive-less path starting with a separator when the host is Windows."""
    if os.name == "nt" and text.startswith(("/", "\\")):
        raise ValueError(
            f"Unix absolute path '{text}' cannot be used on Windows. "
            "Use a drive path such as 'C:/dir/file', a UNC path, or a path relative "
            "to the config file."
        )


def anchor(value: str, ctx: ConfigContext) -> Path:
    """Anchor a config path: substitute placeholders, expand ``~``, join onto the config dir.

    Pure: the filesystem, the working directory and the process environment are never
    read, so symlinks are kept and `a/../b` collapses lexically. ``..`` collapses
    lexically also in absolute paths, so `link/../x` does not follow the symlink `link`.
    Accepts plain paths and ``file:`` / ``sqlite:`` URIs. A Windows absolute path is
    rejected on a posix host, a posix-absolute path on Windows, and a drive-relative
    path (`C:x`) everywhere.
    """
    text = substitute_placeholders(value, ctx)
    if is_windows_absolute_path(text):
        return Path(os.path.normpath(normalize_windows_absolute_path(text)))
    _reject_drive_relative(text)
    if _URI_SCHEME_RE.match(text):
        text = _uri_path(text)
        if is_windows_absolute_path(text.lstrip("/")):
            return Path(os.path.normpath(normalize_windows_absolute_path(text.lstrip("/"))))
    elif (home_path := _expand_home(text, ctx)) is not None:
        return Path(os.path.normpath(home_path))
    _reject_drive_relative(text)
    _reject_posix_absolute_on_windows(text)
    return Path(os.path.normpath(ctx.config_dir / text))


def expand_config_path(value: str, ctx: ConfigContext) -> str:
    """Anchor `value` (see `anchor`) and return it as a string."""
    return str(anchor(value, ctx))


def expand_path_field[T](v: T, info: ValidationInfo, *, glob: bool = False) -> T | str:
    """Shared body for `@field_validator(..., mode="before")` path/glob expansion.

    Args:
        v: Raw field value from Pydantic validation.
        info: Pydantic validation context, carrying the config resolution context.
        glob: If True, expand as a glob pattern via `expand_config_glob`; otherwise
            expand as a plain path via `expand_config_path`.

    Returns:
        The expanded string, or `v` unchanged if it isn't a string or no context is set.
    """
    if info.context is None or not isinstance(v, str):
        return v
    ctx = ConfigContext.from_pydantic_context(info.context)
    return expand_config_glob(v, ctx) if glob else expand_config_path(v, ctx)


def _join_glob_suffix(prefix: str, suffix: str) -> str:
    """Join a resolved path prefix with a glob suffix without reparsing Windows paths."""
    cleaned_prefix = prefix.rstrip("/\\")
    cleaned_suffix = suffix.lstrip("/\\")
    if not cleaned_prefix:
        # An empty prefix stays relative; a bare root ("/") keeps its root.
        return prefix + cleaned_suffix
    separator = "\\" if "\\" in cleaned_prefix and "/" not in cleaned_prefix else "/"
    return f"{cleaned_prefix}{separator}{cleaned_suffix}"


def expand_config_glob(value: str, ctx: ConfigContext) -> str:
    """Like expand_config_path but preserves glob wildcards in the filename.

    Only the prefix up to its last separator is anchored; the rest of the prefix and the
    glob suffix stay verbatim, because anchoring would collapse a trailing `.` or `..`
    (`data/.*` must not become `data*`). Without a separator the prefix is verbatim too
    and joins the config directory. Placeholders are substituted first, so a separator
    inside `${NEURALLS_*}` counts.
    """
    value = substitute_placeholders(value, ctx)
    for i, ch in enumerate(value):
        if ch in _GLOB_CHARS:
            prefix, suffix = value[:i], value[i:]
            split = max(prefix.rfind("/"), prefix.rfind("\\")) + 1
            directory, name = prefix[:split], prefix[split:]
            anchored = expand_config_path(directory, ctx) if directory else str(ctx.config_dir)
            return _join_glob_suffix(anchored, name + suffix)
    return expand_config_path(value, ctx)
