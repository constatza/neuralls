"""Profile-based user configuration stored in the OS user-config directory."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir
from pydantic import BaseModel, Field, field_validator

from neuralls.platform.config.resolution import require_absolute_user_path, resolve_user_path

PROFILE_ENV_VAR = "NEURALLS_PROFILE"
USER_CONFIG_DIR = Path(user_config_dir("neuralls"))
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.toml"
_PROFILE_FIELDS = ("raw_dir", "processed_dir", "output_dir")


class ProfileConfig(BaseModel):
    """Machine-specific root directories stored in the user config.

    Attributes:
        raw_dir: Raw matrices and archives root.
        processed_dir: Processed datasets root.
        output_dir: Outputs (MLflow, checkpoints) root.
    """

    model_config = {"frozen": True}

    raw_dir: Path | None = Field(default=None, description="Raw matrices and archives root.")
    processed_dir: Path = Field(..., description="Processed datasets root.")
    output_dir: Path = Field(..., description="Outputs (MLflow, checkpoints) root.")

    @field_validator("raw_dir", "processed_dir", "output_dir", mode="before")
    @classmethod
    def _normalize_path_input(cls, value: object) -> object:
        """Reject blank path inputs and trim surrounding whitespace."""
        if value is None:
            return value
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Profile paths must not be blank.")
        return stripped

    def to_root_mapping(self) -> dict[str, str]:
        """Return a plain string mapping of the settings-relevant root fields.

        Returns:
            A dict with ``processed_dir`` and ``output_dir`` as strings, plus
            ``raw_dir`` when it is set (used to expand ``${NEURALLS_RAW_DIR}``
            placeholders in dataset configs).
        """
        mapping: dict[str, str] = {
            "processed_dir": str(self.processed_dir),
            "output_dir": str(self.output_dir),
        }
        if self.raw_dir is not None:
            mapping["raw_dir"] = str(self.raw_dir)
        return mapping


def _resolve_config_file(override: Path | None) -> Path:
    return override if override is not None else USER_CONFIG_FILE


def _read_raw(config_file: Path) -> dict[str, Any]:
    with config_file.open("rb") as fh:
        return tomllib.load(fh)


def _write_raw(raw: dict[str, Any], config_file: Path) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open("wb") as fh:
        tomli_w.dump(raw, fh)


def load_profile(
    profile: str | None = None,
    *,
    _config_file: Path | None = None,
) -> ProfileConfig | None:
    """Load a named profile from the user config file.

    Args:
        profile: Profile name to load. Falls back to the ``NEURALLS_PROFILE``
            environment variable, then ``"default"``.
        _config_file: Override config file path, intended for tests only.

    Returns:
        A validated :class:`ProfileConfig`, or ``None`` if no default profile
        exists yet.

    Raises:
        ValueError: If a non-default profile name is requested but not found.
    """
    config_file = _resolve_config_file(_config_file)
    if not config_file.exists():
        return None

    raw = _read_raw(config_file)
    name = profile or os.getenv(PROFILE_ENV_VAR) or "default"

    if name == "default":
        section = raw.get("default")
        if section is None:
            return None
    else:
        section = raw.get("profiles", {}).get(name)
        if section is None:
            raise ValueError(
                f"Profile '{name}' not found in {config_file}. Run: uv run config create {name}"
            )

    return _absolute_profile(ProfileConfig.model_validate(section), name=name)


def map_roots[M: BaseModel](
    model: M, fields: Iterable[str], to_root: Callable[[str, Path], Path]
) -> M:
    """Return *model* with every set root field replaced by ``to_root(field, value)``."""
    return model.model_copy(
        update={
            field: to_root(field, value)
            for field in fields
            if (value := getattr(model, field)) is not None
        }
    )


def resolve_typed_profile(profile: ProfileConfig) -> ProfileConfig:
    """Anchor roots typed on the command line at the working directory, like any CLI path.

    Saved profiles therefore always hold absolute roots, which `load_profile` requires.
    """
    return map_roots(profile, _PROFILE_FIELDS, lambda _field, value: resolve_user_path(value))


def _absolute_profile(profile: ProfileConfig, *, name: str) -> ProfileConfig:
    """Expand ``~`` in every stored root and reject a relative one."""
    return map_roots(
        profile,
        _PROFILE_FIELDS,
        lambda field, value: require_absolute_user_path(value, name=f"profile '{name}' {field}"),
    )


def save_profile(
    name: str,
    config: ProfileConfig,
    *,
    overwrite: bool = False,
    _config_file: Path | None = None,
) -> None:
    """Persist a profile to the user config file.

    Args:
        name: Profile name to save (``"default"`` writes to the top-level
            ``[default]`` section; any other name writes under ``[profiles.<name>]``).
        config: The profile configuration to persist.
        overwrite: If ``False`` (default), raise when the profile already exists.
        _config_file: Override config file path, intended for tests only.

    Raises:
        FileExistsError: If the profile already exists and ``overwrite`` is ``False``.
    """
    config_file = _resolve_config_file(_config_file)
    raw: dict[str, Any] = _read_raw(config_file) if config_file.exists() else {}
    section = {
        field: str(getattr(config, field))
        for field in _PROFILE_FIELDS
        if getattr(config, field) is not None
    }

    if name == "default":
        if "default" in raw and not overwrite:
            raise FileExistsError(
                f"Profile '{name}' already exists. Use overwrite=True to replace."
            )
        raw["default"] = section
    else:
        profiles = raw.setdefault("profiles", {})
        if name in profiles and not overwrite:
            raise FileExistsError(
                f"Profile '{name}' already exists. Use overwrite=True to replace."
            )
        profiles[name] = section

    _write_raw(raw, config_file)


def list_profiles(_config_file: Path | None = None) -> list[str]:
    """List all profile names stored in the user config file.

    Args:
        _config_file: Override config file path, intended for tests only.

    Returns:
        A list of profile names. ``"default"`` appears first if it exists,
        followed by named profiles in alphabetical order.
    """
    config_file = _resolve_config_file(_config_file)
    if not config_file.exists():
        return []

    raw = _read_raw(config_file)
    names: list[str] = []
    if "default" in raw:
        names.append("default")
    names.extend(sorted(raw.get("profiles", {}).keys()))
    return names


def delete_profile(name: str, _config_file: Path | None = None) -> None:
    """Remove a named profile from the user config file.

    Args:
        name: Profile name to delete.
        _config_file: Override config file path, intended for tests only.

    Raises:
        ValueError: If ``name`` is ``"default"``.
        FileNotFoundError: If the config file does not exist.
        KeyError: If the named profile does not exist.
    """
    if name == "default":
        raise ValueError("Cannot delete the default profile.")

    config_file = _resolve_config_file(_config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"No config file at {config_file}.")

    raw = _read_raw(config_file)
    if name not in raw.get("profiles", {}):
        raise KeyError(f"Profile '{name}' not found.")

    del raw["profiles"][name]
    if not raw["profiles"]:
        del raw["profiles"]

    _write_raw(raw, config_file)
