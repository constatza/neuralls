"""The absolute case config path: one spelling for session tags and report lookups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConfigKey:
    """One case config's identity, ``<case>/<config>``: the name of its MLflow experiment."""

    case: str
    config: str

    @property
    def experiment_name(self) -> str:
        return f"{self.case}/{self.config}"


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

    @property
    def key(self) -> ConfigKey:
        """The case directory and the config file's stem: one experiment per config."""
        return ConfigKey(case=self.path.parent.name, config=self.path.stem)
