"""Pure path and MLflow resolution policy for neuralls."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

from dlkit.infrastructure.io import url_resolver

from neuralls.shared.constants import DEFAULT_PROJECT_ROOT

CASE_CONFIG_ENV_VAR = "NEURALLS_CASE_CONFIG"
ENV_FILE_ENV_VAR = "NEURALLS_ENV_FILE"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC_RE = re.compile(r"^(?:\\\\|//)[^\\/]+[\\/][^\\/]+")


@dataclass(frozen=True, slots=True)
class PathContext:
    """Resolved base paths for an experiment."""

    project_root: Path
    output_root: Path
    processed_root: Path


@dataclass(frozen=True, slots=True)
class MlflowPaths:
    """Resolved MLflow URIs."""

    tracking_uri: str
    artifact_uri: str | None


def is_windows_absolute_path(value: Path | str) -> bool:
    """Return True for Windows drive-letter and UNC absolute paths."""
    text = str(value).strip()
    return bool(_WINDOWS_DRIVE_RE.match(text) or _WINDOWS_UNC_RE.match(text))


def normalize_windows_absolute_path(value: Path | str) -> Path:
    """Normalize a Windows absolute path or reject it on non-Windows hosts."""
    normalized = PureWindowsPath(str(value))
    if os.name == "nt":
        return Path(str(normalized))
    raise ValueError(
        f"Windows absolute path '{normalized.as_posix()}' cannot be used on posix platform. "
        "Use relative paths or platform-specific configuration."
    )


def resolve_local_path(
    value: Path | str,
    *,
    base_dir: Path,
) -> Path:
    """Resolve a local path or local URI against a base directory."""
    if is_windows_absolute_path(value):
        return normalize_windows_absolute_path(value)
    return url_resolver.resolve_local_uri(str(value), base_dir.resolve())


def build_sqlite_uri(path: Path) -> str:
    """Build a normalized sqlite URI from a local database path."""
    return url_resolver.build_uri(path.resolve(), scheme="sqlite")


def build_sqlite_tracking_uri(db_path: Path) -> str:
    """Build a normalized sqlite tracking URI from a local DB path."""
    return build_sqlite_uri(db_path)


def is_sqlite_tracking_uri(uri: str) -> bool:
    """Return True when tracking URI uses the sqlite scheme."""
    return url_resolver.scheme(uri.strip()) == "sqlite"


def expand_user_path(value: Path | str) -> Path:
    """Expand ``~`` using the platform's home-directory resolution."""
    return Path(value).expanduser()


def resolve_user_path(value: Path | str) -> Path:
    """Expand ``~`` and resolve to an absolute local path."""
    expanded = expand_user_path(value)
    if is_windows_absolute_path(expanded):
        return normalize_windows_absolute_path(expanded)
    return expanded.resolve()


def require_absolute_user_path(value: Path | str, *, name: str) -> Path:
    """Expand ``~`` and require an absolute path, naming the setting when it is relative.

    Values from environment variables and the stored profile have no file to anchor
    against, and anchoring them at the working directory makes outputs move with ``cd``.
    """
    expanded = expand_user_path(value)
    if is_windows_absolute_path(expanded):
        return normalize_windows_absolute_path(expanded)
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be an absolute path (or start with ~), got {str(value)!r}.")
    return expanded


def resolve_case_config_path(case_config_path: Path | None) -> Path | None:
    """Resolve an explicit case config path or the env override."""
    if case_config_path is not None:
        return resolve_user_path(case_config_path)
    configured_case = os.getenv(CASE_CONFIG_ENV_VAR)
    if configured_case is None or not configured_case.strip():
        return None
    return resolve_user_path(configured_case)


def resolve_env_file_path(env_file: Path | None) -> Path | None:
    """Resolve an explicit env file path or the env override."""
    if env_file is not None:
        return resolve_user_path(env_file)
    configured_env_file = os.getenv(ENV_FILE_ENV_VAR)
    if configured_env_file is None or not configured_env_file.strip():
        return None
    return resolve_user_path(configured_env_file)


def resolve_registry_path(base_dir: Path, relative_or_absolute: Path | str) -> Path:
    """Resolve one registry path against a config directory."""
    return resolve_local_path(relative_or_absolute, base_dir=base_dir)


def resolve_path_context(
    *,
    processed_root: Path | str,
    output_root: Path | str,
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    data_dir_override: Path | str | None = None,
    output_override: Path | str | None = None,
) -> PathContext:
    """Pick experiment roots into a ``PathContext``: overrides win, paths kept as given.

    Every input is already absolute where it comes from disk: settings roots by their
    loaders, ``data_dir`` by the config validators, the output override by its caller.
    """
    return PathContext(
        project_root=Path(project_root),
        output_root=Path(output_override or output_root),
        processed_root=Path(data_dir_override or processed_root),
    )


def normalize_tracking_uri(tracking_uri: str, *, base_dir: Path) -> str:
    """Validate and normalize a runtime MLflow tracking URI."""
    uri = tracking_uri.strip()
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https", "sqlite"}:
        raise ValueError("MLflow tracking_uri must use one of: http://, https://, sqlite:///.")

    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError(
                f"MLflow tracking_uri must include a host for {parsed.scheme} URIs: {uri!r}"
            )
        return uri.rstrip("/")

    db_path = _resolve_sqlite_db_path(uri, base_dir=base_dir)
    return build_sqlite_tracking_uri(db_path)


def normalize_artifacts_destination(artifacts_destination: str, *, base_dir: Path) -> str:
    """Normalize an artifacts destination to an absolute path anchored at ``base_dir``."""
    return str(resolve_local_path(artifacts_destination, base_dir=base_dir))


def derive_sqlite_artifacts_destination(tracking_uri: str, *, base_dir: Path) -> str:
    """Derive artifacts destination as a sibling ``mlartifacts`` directory."""
    normalized = normalize_tracking_uri(tracking_uri, base_dir=base_dir)
    if not is_sqlite_tracking_uri(normalized):
        raise ValueError(
            f"Cannot derive sqlite artifacts destination from non-sqlite URI: {tracking_uri!r}"
        )
    db_path = _resolve_sqlite_db_path(normalized, base_dir=base_dir)
    output_root = _derive_output_root_from_db_path(db_path)
    return str((output_root / "mlartifacts").resolve())


def derive_output_root_from_tracking_uri(tracking_uri: str, *, base_dir: Path) -> Path | None:
    """Derive output root from a sqlite tracking URI, if applicable."""
    normalized = normalize_tracking_uri(tracking_uri, base_dir=base_dir)
    if not is_sqlite_tracking_uri(normalized):
        return None
    db_path = _resolve_sqlite_db_path(normalized, base_dir=base_dir)
    return _derive_output_root_from_db_path(db_path)


def to_mlflow_artifact_location(value: str) -> str:
    """Convert an absolute local artifact root into an MLflow-compatible location."""
    stripped = value.strip()
    if url_resolver.scheme(stripped):
        return stripped
    if is_windows_absolute_path(stripped):
        return url_resolver.build_uri(normalize_windows_absolute_path(stripped), scheme="file")
    path = Path(stripped)
    if not path.is_absolute():
        raise ValueError(f"MLflow artifact location must be absolute, got {stripped!r}.")
    return url_resolver.build_uri(path, scheme="file")


def build_mlflow_environment(
    *,
    tracking_uri: str,
    artifacts_destination: str | None = None,
    base_dir: Path,
) -> dict[str, str]:
    """Build validated MLflow env vars; relative local values anchor at ``base_dir``."""
    normalized_tracking = normalize_tracking_uri(tracking_uri, base_dir=base_dir)
    if artifacts_destination is None:
        if not is_sqlite_tracking_uri(normalized_tracking):
            return {"MLFLOW_TRACKING_URI": normalized_tracking}
        normalized_artifacts = derive_sqlite_artifacts_destination(
            normalized_tracking, base_dir=base_dir
        )
    else:
        normalized_artifacts = normalize_artifacts_destination(
            artifacts_destination, base_dir=base_dir
        )

    return {
        "MLFLOW_TRACKING_URI": normalized_tracking,
        "MLFLOW_ARTIFACT_URI": normalized_artifacts,
    }


def _resolve_sqlite_db_path(tracking_uri: str, *, base_dir: Path) -> Path:
    """Resolve sqlite DB path from tracking URI to an absolute local path."""
    if not is_sqlite_tracking_uri(tracking_uri):
        raise ValueError(f"Expected sqlite tracking URI, got: {tracking_uri!r}")

    uri = tracking_uri.strip()
    if uri.endswith(":memory:"):
        raise ValueError("MLflow sqlite tracking URIs must include a filesystem database path.")
    return url_resolver.resolve_local_uri(uri, base_dir.resolve())


def _derive_output_root_from_db_path(db_path: Path) -> Path:
    mlruns_dir = db_path.parent.resolve()
    if mlruns_dir.name == "mlruns":
        return mlruns_dir.parent
    return mlruns_dir
