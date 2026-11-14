"""The absolute case config path: one spelling for session tags and report lookups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CaseConfigPath:
    """The absolute path of one case config: the single spelling session tags and report lookups use."""

    path: Path

    def __post_init__(self) -> None:
        """Reject a relative path: a relative spelling would break report lookup."""
        if not self.path.is_absolute():
            raise ValueError(f"Case config path must be absolute, got {self.path}")

    @classmethod
    def resolve(cls, path: Path) -> CaseConfigPath:
        """Build from user input: the CLI's (and tests') one constructor."""
        return cls(path.resolve())

    @property
    def config_dir(self) -> Path:
        return self.path.parent
