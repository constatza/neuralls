"""In-memory ``ArtifactSink`` keeping the bytes of every file a report logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InMemoryArtifactSink:
    """Stores each logged file's content at the time of logging, by directory and file name."""

    files: dict[tuple[str | None, str], bytes] = field(default_factory=dict)

    def log(self, local_path: Path, directory: str | None) -> None:
        self.files[directory, local_path.name] = local_path.read_bytes()
