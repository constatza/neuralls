"""Dataset identity resolution helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from neuralls.platform.config.models.data_models import DataConfigFile

_MLFLOW_ALIAS_PATTERN = re.compile(r"^[\w\-]+$")


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Resolved dataset identity with source traceability."""

    name: str
    registry_alias: str
    source: Literal["id"]


def _coerce_non_empty_str(value: Any) -> str | None:
    """Return stripped string value when non-empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def normalize_registry_id(alias: str) -> str:
    """Normalize user-facing registry id or alias to MLflow-safe form.

    Supports optional ``@`` prefix for user-facing references.
    """
    normalized = alias.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:].strip()
    if not normalized:
        raise ValueError("Registry alias cannot be empty.")
    if not _MLFLOW_ALIAS_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Registry alias must match ^[\\w\\-]+$ (letters, numbers, underscore, hyphen)."
        )
    return normalized


def _read_legacy_dataset_name(value: Any) -> str | None:
    """Read a legacy [dataset].name fallback when present."""
    if not isinstance(value, Mapping):
        return None
    return _coerce_non_empty_str(value.get("name"))


def _require_dataset_name(
    configured_dataset: str | None,
    *,
    config_path: Path | str | None,
    legacy_dataset: Any = None,
) -> str:
    """Resolve dataset identity from top-level id, legacy section, or filename."""
    if configured_dataset is not None:
        return configured_dataset
    legacy_name = _read_legacy_dataset_name(legacy_dataset)
    if legacy_name is not None:
        return legacy_name
    if config_path is not None:
        return Path(config_path).stem
    raise ValueError(
        "Missing required top-level 'id' in data config. Dataset name must be explicitly defined."
    )


def resolve_dataset_identity(
    *,
    data_cfg: DataConfigFile,
    config_path: Path | str | None = None,
) -> DatasetIdentity:
    """Resolve dataset identity from validated data config + config path."""
    configured = _coerce_non_empty_str(getattr(data_cfg, "id", None))
    dataset_name = _require_dataset_name(
        configured,
        config_path=config_path,
        legacy_dataset=getattr(data_cfg, "dataset", None),
    )
    return DatasetIdentity(
        name=dataset_name,
        registry_alias=normalize_registry_id(dataset_name),
        source="id",
    )


def resolve_dataset_identity_from_mapping(
    *,
    config: Mapping[str, Any],
    config_path: Path | str | None = None,
) -> DatasetIdentity:
    """Resolve dataset identity from a raw config mapping.

    Raw mappings stay strict: callers must provide an explicit top-level ``id``.
    """
    _ = config_path
    configured = _coerce_non_empty_str(config.get("id"))
    dataset_name = _require_dataset_name(configured, config_path=None)
    return DatasetIdentity(
        name=dataset_name,
        registry_alias=normalize_registry_id(dataset_name),
        source="id",
    )
