"""Minimal artifact lease manager that maps artifact paths onto fixed local roots."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from neuralls.platform.tracking.artifact_access import ArtifactLease


class CheckpointLeaseManager:
    """Leases `checkpoints` from one local root and every other artifact path from a fallback root."""

    def __init__(self, checkpoint_root: Path, fallback_root: Path | None = None) -> None:
        self._checkpoint_root = checkpoint_root
        self._fallback_root = fallback_root or checkpoint_root
        self.file_calls: list[tuple[str, str]] = []
        self.dir_calls: list[tuple[str, str]] = []

    @property
    def tracking_uri(self) -> str | None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.file_calls.append((run_id, artifact_path))
        return ArtifactLease(
            path=self._fallback_root / artifact_path,
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri="file:///artifacts",
            local_copy=False,
        )

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.dir_calls.append((run_id, artifact_path))
        root = self._checkpoint_root if artifact_path == "checkpoints" else self._fallback_root
        return ArtifactLease(
            path=root,
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri="file:///artifacts",
            local_copy=False,
        )
