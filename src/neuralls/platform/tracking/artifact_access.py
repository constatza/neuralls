"""Lease-backed access to MLflow run artifacts.

The public contract of this module is intentionally small: callers ask for a
run artifact as a local file or directory path, and the lease manager decides
whether that path can be borrowed directly from a local artifact store or must
be materialized into scoped scratch storage.
"""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Literal, Protocol, Self

from dlkit.infrastructure.io import url_resolver
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.platform.config.resolution import resolve_local_path

type ArtifactKind = Literal["file", "dir"]
type DownloadArtifact = Callable[..., str]

ARTIFACT_KIND_FILE: ArtifactKind = "file"
ARTIFACT_KIND_DIR: ArtifactKind = "dir"
LOCAL_ARTIFACT_SCHEMES = frozenset({"", "file"})
ARTIFACT_PATH_SEPARATOR = "/"
UNSAFE_ARTIFACT_PATH_SEGMENTS = frozenset({"", ".", ".."})
INVALID_ARTIFACT_PATH_CHARS = frozenset({"\\"})
REMOTE_ARTIFACT_DOWNLOAD_PREFIX = "neuralls-mlflow-artifacts-"


@dataclass(frozen=True)
class ArtifactLease:
    """A local path to an MLflow artifact with explicit ownership metadata."""

    path: Path
    run_id: str
    artifact_path: str
    source_uri: str
    local_copy: bool


class ArtifactLeaseManager(Protocol):
    """Resolve MLflow artifact references to local paths for a managed lifetime."""

    @property
    @abstractmethod
    def tracking_uri(self) -> str | None:
        """Tracking URI associated with this lease manager, if any."""
        ...

    @abstractmethod
    def __enter__(self) -> Self:
        """Enter the lease scope."""
        ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release any materialized artifact copies."""
        ...

    @abstractmethod
    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Return a lease to a local artifact file."""
        ...

    @abstractmethod
    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Return a lease to a local artifact directory."""
        ...


class NoopArtifactLeaseManager:
    """Lease manager for code paths that must not resolve MLflow artifacts."""

    @property
    def tracking_uri(self) -> str | None:
        """No-op managers are not bound to a tracking store."""
        return None

    def __enter__(self) -> Self:
        """Enter a no-op lease scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit a no-op lease scope."""
        del exc_type, exc, tb

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Fail because this manager is only valid for non-resolving paths."""
        raise RuntimeError(_noop_resolution_error(run_id=run_id, artifact_path=artifact_path))

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Fail because this manager is only valid for non-resolving paths."""
        raise RuntimeError(_noop_resolution_error(run_id=run_id, artifact_path=artifact_path))


class MlflowArtifactLeaseManager:
    """Resolve MLflow artifacts while avoiding unnecessary local copies."""

    def __init__(
        self,
        *,
        client: MlflowClient,
        download_artifact: DownloadArtifact | None = None,
    ) -> None:
        self._client = client
        self._tracking_uri = str(client.tracking_uri)
        self._download_artifact = download_artifact or client.download_artifacts
        self._temporary_dirs: list[tempfile.TemporaryDirectory[str]] = []
        self._leases: dict[tuple[str, str, ArtifactKind], ArtifactLease] = {}
        self._entered = False

    @property
    def tracking_uri(self) -> str | None:
        """Tracking URI associated with the MLflow client."""
        return self._tracking_uri

    def __enter__(self) -> Self:
        """Enter the artifact lease scope."""
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release materialized artifact copies owned by this manager."""
        del exc_type, exc, tb
        for temporary_dir in reversed(self._temporary_dirs):
            temporary_dir.cleanup()
        self._temporary_dirs.clear()
        self._leases.clear()
        self._entered = False

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Return a lease to a local artifact file."""
        return self._resolve(run_id=run_id, artifact_path=artifact_path, kind=ARTIFACT_KIND_FILE)

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        """Return a lease to a local artifact directory."""
        return self._resolve(run_id=run_id, artifact_path=artifact_path, kind=ARTIFACT_KIND_DIR)

    def _resolve(self, *, run_id: str, artifact_path: str, kind: ArtifactKind) -> ArtifactLease:
        if not self._entered:
            raise RuntimeError("Artifact leases must be resolved inside a lease-manager context.")
        normalized_path = normalize_artifact_path(artifact_path)
        cache_key = (run_id, normalized_path, kind)
        if cache_key in self._leases:
            logger.info(
                "Reusing materialized MLflow artifact '{}' from run {}.",
                normalized_path,
                run_id,
            )
            return self._leases[cache_key]

        logger.info("Resolving artifact-store URI for MLflow run {}.", run_id)
        source_uri = _run_artifact_uri(self._client, run_id=run_id)
        logger.info("Resolved artifact-store URI for MLflow run {}.", run_id)
        path, local_copy = self._materialize_path(
            run_id=run_id,
            artifact_path=normalized_path,
            source_uri=source_uri,
        )
        logger.info(
            "Validating materialized MLflow artifact '{}' from run {}.",
            normalized_path,
            run_id,
        )
        _validate_materialized_path(path, kind=kind, run_id=run_id, artifact_path=normalized_path)
        if local_copy:
            _validate_temporary_lease_path(path, self._temporary_dirs[-1])
        lease = ArtifactLease(
            path=path,
            run_id=run_id,
            artifact_path=normalized_path,
            source_uri=source_uri,
            local_copy=local_copy,
        )
        self._leases[cache_key] = lease
        return lease

    def _materialize_path(
        self,
        *,
        run_id: str,
        artifact_path: str,
        source_uri: str,
    ) -> tuple[Path, bool]:
        if _is_local_artifact_uri(source_uri):
            return _local_artifact_path(source_uri=source_uri, artifact_path=artifact_path), False

        temporary_dir = tempfile.TemporaryDirectory(prefix=REMOTE_ARTIFACT_DOWNLOAD_PREFIX)
        try:
            logger.info(
                "Starting MLflow artifact download '{}' from run {}.",
                artifact_path,
                run_id,
            )
            downloaded_path = self._download_artifact(
                run_id=run_id,
                path=artifact_path,
                dst_path=temporary_dir.name,
            )
            logger.info(
                "MLflow artifact download '{}' from run {} returned; resolving local path.",
                artifact_path,
                run_id,
            )
            path = Path(downloaded_path).resolve()
            logger.info(
                "Resolved local path for MLflow artifact '{}' from run {}.",
                artifact_path,
                run_id,
            )
        except Exception:
            temporary_dir.cleanup()
            raise
        self._temporary_dirs.append(temporary_dir)
        return path, True


def normalize_artifact_path(value: str) -> str:
    """Normalize and validate an MLflow artifact-relative path."""
    raw = value.strip()
    _validate_raw_artifact_path(raw)
    _validate_artifact_path_segments(
        tuple(raw.split(ARTIFACT_PATH_SEPARATOR)),
        original=value,
    )
    path = PurePosixPath(raw)
    _validate_posix_artifact_path(path)
    _validate_artifact_path_chars(raw, original=value)
    return path.as_posix()


def _noop_resolution_error(*, run_id: str, artifact_path: str) -> str:
    return (
        "No-op artifact lease manager cannot resolve MLflow artifacts "
        f"for run '{run_id}' at '{artifact_path}'."
    )


def _validate_raw_artifact_path(raw: str) -> None:
    if raw and not raw.startswith("/"):
        return
    raise ValueError("artifact_path must be a non-empty relative path.")


def _validate_posix_artifact_path(path: PurePosixPath) -> None:
    if not path.is_absolute():
        return
    raise ValueError("artifact_path must be a non-empty relative path.")


def _validate_artifact_path_chars(raw: str, *, original: str) -> None:
    if not any(char in raw for char in INVALID_ARTIFACT_PATH_CHARS):
        return
    raise ValueError(f"Unsafe artifact_path: {original!r}")


def _validate_artifact_path_segments(parts: tuple[str, ...], *, original: str) -> None:
    if not any(part in UNSAFE_ARTIFACT_PATH_SEGMENTS for part in parts):
        return
    raise ValueError(f"Unsafe artifact_path: {original!r}")


def _run_artifact_uri(client: MlflowClient, *, run_id: str) -> str:
    run = client.get_run(run_id)
    artifact_uri = getattr(getattr(run, "info", None), "artifact_uri", None)
    match artifact_uri:
        case str() as uri if uri.strip():
            return uri
        case _:
            raise RuntimeError(f"MLflow run '{run_id}' has no artifact_uri.")


def _is_local_artifact_uri(uri: str) -> bool:
    scheme = url_resolver.scheme(uri.strip())
    return scheme in LOCAL_ARTIFACT_SCHEMES


def _local_artifact_path(*, source_uri: str, artifact_path: str) -> Path:
    root = resolve_local_path(source_uri, base_dir=Path.cwd())
    artifact_parts = PurePosixPath(artifact_path).parts
    return root.joinpath(*artifact_parts).resolve()


def _validate_materialized_path(
    path: Path,
    *,
    kind: ArtifactKind,
    run_id: str,
    artifact_path: str,
) -> None:
    match kind:
        case kind_value if kind_value == ARTIFACT_KIND_FILE and not path.is_file():
            raise FileNotFoundError(
                f"Expected MLflow artifact file for run '{run_id}' at "
                f"'{artifact_path}', got: {path}"
            )
        case kind_value if kind_value == ARTIFACT_KIND_DIR and not path.is_dir():
            raise FileNotFoundError(
                f"Expected MLflow artifact directory for run '{run_id}' at "
                f"'{artifact_path}', got: {path}"
            )
        case _:
            return


def _validate_temporary_lease_path(
    path: Path,
    temporary_dir: tempfile.TemporaryDirectory[str],
) -> None:
    if path.is_relative_to(Path(temporary_dir.name).resolve()):
        return
    raise RuntimeError(f"MLflow returned artifact path outside the managed scratch dir: {path}")


__all__ = [
    "ArtifactLease",
    "ArtifactLeaseManager",
    "MlflowArtifactLeaseManager",
    "normalize_artifact_path",
]
