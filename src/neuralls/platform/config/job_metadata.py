"""Read stable identity metadata from lower-case DLKit job files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class JobMetadata:
    """Identity fields resolved from one lower-case job and its model profile."""

    experiment_name: str | None = None
    model_name: str | None = None


def _load_toml(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError, OSError, tomllib.TOMLDecodeError, ValueError:
        return None
    return raw if isinstance(raw, dict) else None


def _read_str(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _read_model_name(raw: dict[str, object]) -> str | None:
    model_section = raw.get("model")
    if not isinstance(model_section, dict):
        return None
    normalized = dict(model_section)
    return _read_str(normalized, "name") or _read_str(normalized, "class")


def read_job_metadata(job_config_path: Path) -> JobMetadata:
    """Resolve experiment and model identity from one lower-case DLKit job."""
    raw = _load_toml(job_config_path)
    if raw is None:
        return JobMetadata()

    experiment_name = None
    experiment_section = raw.get("experiment")
    if isinstance(experiment_section, dict):
        experiment_name = _read_str(cast(dict[str, object], experiment_section), "name")

    model_name = _read_model_name(raw)
    if model_name is not None:
        return JobMetadata(experiment_name=experiment_name, model_name=model_name)

    run_section = raw.get("run")
    if not isinstance(run_section, dict):
        return JobMetadata(experiment_name=experiment_name)

    model_ref = _read_str(cast(dict[str, object], run_section), "model")
    if model_ref is None:
        return JobMetadata(experiment_name=experiment_name)

    model_profile_raw = _load_toml((job_config_path.parent / model_ref).resolve())
    if model_profile_raw is None:
        return JobMetadata(experiment_name=experiment_name)

    return JobMetadata(
        experiment_name=experiment_name,
        model_name=_read_model_name(model_profile_raw),
    )
