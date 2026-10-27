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

    @property
    def mlflow_tracking_uri(self) -> str:
        """SQLite tracking database URI derived from ``output_root``."""
        return build_sqlite_tracking_uri(self.output_root / "mlruns" / "mlflow.db")

    @property
    def mlflow_artifact_location(self) -> str:
        """MLflow artifact storage location derived from ``output_root``."""
        return str((self.output_root / "mlartifacts").resolve())


@dataclass(frozen=True, slots=True)
class MlflowPaths:
    """Resolved MLflow URIs."""

    tracking_uri: str
    artifact_uri: str | None


def _is_windows_absolute_path(value: Path | str) -> bool:
    """Return True for Windows drive-letter and UNC absolute paths."""
    text = str(value).strip()
    return bool(_WINDOWS_DRIVE_RE.match(text) or _WINDOWS_UNC_RE.match(text))


def _normalize_windows_absolute_path(value: Path | str) -> Path:
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
    if _is_windows_absolute_path(value):
        return _normalize_windows_absolute_path(value)
    return url_resolver.resolve_local_uri(str(value), base_dir.resolve())


def resolve_optional_local_path(
    value: Path | str | None,
    *,
    base_dir: Path,
) -> Path | None:
    """Resolve an optional local path value when present."""
    if value is None:
        return None
    return resolve_local_path(value, base_dir=base_dir)


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
    if _is_windows_absolute_path(expanded):
        return _normalize_windows_absolute_path(expanded)
    return expanded.resolve()


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
    """Resolve experiment roots into a ``PathContext`` value object."""
    resolved_output_root = resolve_user_path(output_override or output_root)
    resolved_processed_root = resolve_user_path(data_dir_override or processed_root)
    return PathContext(
        project_root=resolve_user_path(project_root),
        output_root=resolved_output_root,
        processed_root=resolved_processed_root,
    )


def normalize_tracking_uri(
    tracking_uri: str,
    *,
    config_path: Path | None = None,
) -> str:
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

    db_path = _resolve_sqlite_db_path(uri, config_path=config_path)
    return build_sqlite_tracking_uri(db_path)


def normalize_artifacts_destination(
    artifacts_destination: str,
    *,
    config_path: Path | None = None,
) -> str:
    """Normalize an artifacts destination to an absolute filesystem path."""
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    return str(resolve_local_path(artifacts_destination, base_dir=base_dir))


def derive_sqlite_artifacts_destination(tracking_uri: str) -> str:
    """Derive artifacts destination as a sibling ``mlartifacts`` directory."""
    normalized = normalize_tracking_uri(tracking_uri)
    if not is_sqlite_tracking_uri(normalized):
        raise ValueError(
            f"Cannot derive sqlite artifacts destination from non-sqlite URI: {tracking_uri!r}"
        )
    db_path = _resolve_sqlite_db_path(normalized)
    output_root = _derive_output_root_from_db_path(db_path)
    return str((output_root / "mlartifacts").resolve())


def derive_output_root_from_tracking_uri(
    tracking_uri: str,
    *,
    config_path: Path | None = None,
) -> Path | None:
    """Derive output root from a sqlite tracking URI, if applicable."""
    normalized = normalize_tracking_uri(tracking_uri, config_path=config_path)
    if not is_sqlite_tracking_uri(normalized):
        return None
    db_path = _resolve_sqlite_db_path(normalized)
    return _derive_output_root_from_db_path(db_path)


def to_mlflow_artifact_location(
    value: str,
    *,
    base_dir: Path | None = None,
) -> str:
    """Convert a local artifact root into an MLflow-compatible location."""
    stripped = value.strip()
    if url_resolver.scheme(stripped):
        return stripped
    anchor = base_dir.resolve() if base_dir is not None else Path.cwd().resolve()
    return url_resolver.build_uri(resolve_local_path(stripped, base_dir=anchor), scheme="file")


def build_mlflow_environment(
    *,
    tracking_uri: str,
    artifacts_destination: str | None = None,
    config_path: Path | None = None,
) -> dict[str, str]:
    """Build validated MLflow env vars for runtime execution."""
    normalized_tracking = normalize_tracking_uri(
        tracking_uri,
        config_path=config_path,
    )
    if artifacts_destination is None:
        if not is_sqlite_tracking_uri(normalized_tracking):
            return {"MLFLOW_TRACKING_URI": normalized_tracking}
        normalized_artifacts = derive_sqlite_artifacts_destination(normalized_tracking)
    else:
        normalized_artifacts = normalize_artifacts_destination(
            artifacts_destination,
            config_path=config_path,
        )

    return {
        "MLFLOW_TRACKING_URI": normalized_tracking,
        "MLFLOW_ARTIFACT_URI": normalized_artifacts,
    }


def resolve_mlflow_paths(
    *,
    tracking_uri: str | None,
    artifact_uri: str | None,
    project_root: Path | str | None = None,
    workspace_root: Path | str | None = None,
    default_tracking_uri: str | None = None,
    default_artifact_location: str | None = None,
) -> MlflowPaths:
    """Resolve tracking and artifact URIs against explicit defaults and roots."""
    resolved_tracking_uri = _resolve_tracking_uri(
        tracking_uri=tracking_uri,
        project_root=project_root,
        default_tracking_uri=default_tracking_uri,
    )
    resolved_workspace_root = (
        resolve_user_path(workspace_root) if workspace_root is not None else None
    )
    resolved_artifact_uri = _resolve_artifact_uri(
        artifact_uri=artifact_uri,
        workspace_root=resolved_workspace_root,
        default_artifact_location=default_artifact_location,
    )
    return MlflowPaths(
        tracking_uri=resolved_tracking_uri,
        artifact_uri=resolved_artifact_uri,
    )


def _resolve_sqlite_db_path(
    tracking_uri: str,
    *,
    config_path: Path | None = None,
) -> Path:
    """Resolve sqlite DB path from tracking URI to an absolute local path."""
    if not is_sqlite_tracking_uri(tracking_uri):
        raise ValueError(f"Expected sqlite tracking URI, got: {tracking_uri!r}")

    uri = tracking_uri.strip()
    if uri.endswith(":memory:"):
        raise ValueError("MLflow sqlite tracking URIs must include a filesystem database path.")
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    return url_resolver.resolve_local_uri(uri, base_dir.resolve())


def _derive_output_root_from_db_path(db_path: Path) -> Path:
    mlruns_dir = db_path.parent.resolve()
    if mlruns_dir.name == "mlruns":
        return mlruns_dir.parent
    return mlruns_dir


def _resolve_tracking_uri(
    *,
    tracking_uri: str | None,
    project_root: Path | str | None,
    default_tracking_uri: str | None,
) -> str:
    if tracking_uri:
        config_path = None
        if project_root is not None:
            config_path = resolve_user_path(project_root) / "config.toml"
        return normalize_tracking_uri(tracking_uri, config_path=config_path)
    if default_tracking_uri:
        return default_tracking_uri
    raise ValueError("tracking_uri is required when no default tracking URI is provided.")


def _resolve_artifact_uri(
    *,
    artifact_uri: str | None,
    workspace_root: Path | None,
    default_artifact_location: str | None,
) -> str:
    if artifact_uri:
        return _normalize_artifact_uri(artifact_uri, workspace_root=workspace_root)
    if default_artifact_location:
        return _normalize_artifact_uri(
            default_artifact_location,
            workspace_root=workspace_root,
        )
    if workspace_root is None:
        raise ValueError(
            "workspace_root is required when no explicit or default artifact location is provided."
        )
    return str((workspace_root / "mlartifacts").resolve())


def _normalize_artifact_uri(
    artifact_uri: str,
    *,
    workspace_root: Path | None,
) -> str:
    scheme = url_resolver.scheme(artifact_uri.strip())
    if scheme and scheme != "file":
        return artifact_uri.strip().rstrip("/")
    if workspace_root is None:
        raise ValueError("workspace_root is required when resolving local artifact locations.")
    return str(resolve_local_path(artifact_uri, base_dir=workspace_root).resolve())
