"""TOML loading and validation for neuralls-owned config types."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    parse_comparison_config,
)
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.models.experiments import CaseConfig, SharedTrackingSettings
from neuralls.platform.config.settings import NeurallsSettings, load_case_settings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT

_DEFAULT_TRACKING_TOML = DEFAULT_PROJECT_ROOT / "configs" / "tracking.toml"


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dict without validation."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_data_config(path: Path, settings: NeurallsSettings) -> DataConfigFile:
    """Load and validate a data config TOML using context-aware path expansion."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    return DataConfigFile.model_validate(raw, context=ctx.as_pydantic_context())


def load_comparison_config(path: Path, settings: NeurallsSettings) -> ComparisonConfig:
    """Load and validate a comparison TOML using context-aware path expansion."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    return parse_comparison_config(raw, context=ctx)


def _fill_missing_dataset_ids(raw: dict[str, Any], ctx: ConfigContext) -> None:
    """Default missing [[datasets]] entry ids from their referenced dataset config's own id.

    A [[datasets]] entry's ``id`` is only a case-registry lookup key; the dataset
    config's own ``id`` is what generation actually names the processed directory
    with. Reading it here (instead of requiring it hand-typed twice) makes the
    dataset config the single source of truth for entries that don't need a
    distinct local alias. Mutates ``raw["datasets"]`` in place; entries with an
    explicit non-blank id are left untouched.
    """
    datasets = raw.get("datasets")
    if not isinstance(datasets, list):
        return
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        existing_id = entry.get("id")
        if isinstance(existing_id, str) and existing_id.strip():
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        dataset_path = Path(expand_config_path(raw_path, ctx))
        dataset_raw = load_raw_toml(dataset_path)
        dataset_id = dataset_raw.get("id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(
                f"[[datasets]] entry with path '{raw_path}' has no 'id' and its dataset "
                f"config at '{dataset_path}' also has no 'id'. Set one explicitly."
            )
        entry["id"] = dataset_id


def _render_dataset_sweep_entries(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one [[dataset_sweeps]] entry into [[datasets]]-shaped dicts (id left unset).

    Pure — no I/O, no context. Each returned dict has only 'path' (and
    'display_name' when a template is given); 'id' is deliberately absent so
    the caller's subsequent _fill_missing_dataset_ids pass resolves it from
    the real dataset config it points at, the same way a hand-written
    [[datasets]] entry without an id would be resolved.
    """
    label = sweep.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("[[dataset_sweeps]] entry is missing a non-blank 'label'.")
    path_template = sweep.get("path_template")
    if not isinstance(path_template, str) or "{value}" not in path_template:
        raise ValueError(
            f"[[dataset_sweeps]] entry '{label}' has no 'path_template' containing '{{value}}'."
        )
    values = sweep.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"[[dataset_sweeps]] entry '{label}' has no non-empty 'values' list.")
    display_name_template = sweep.get("display_name_template")

    entries: list[dict[str, Any]] = []
    for value in values:
        entry: dict[str, Any] = {"path": path_template.replace("{value}", str(value))}
        if isinstance(display_name_template, str):
            entry["display_name"] = display_name_template.replace("{value}", str(value))
        entries.append(entry)
    return entries


def _expand_dataset_sweeps(
    raw: dict[str, Any], ctx: ConfigContext
) -> dict[str, list[dict[str, Any]]]:
    """Expand raw["dataset_sweeps"] into raw["datasets"] entries; return {label: [entries]}.

    Mutates raw in place: pops 'dataset_sweeps' and appends generated entries
    to raw["datasets"]. The returned dicts are the exact same objects
    appended to raw["datasets"], so a later _fill_missing_dataset_ids(raw, ctx)
    call resolves their 'id' in place and this function's caller can read it
    back through the returned mapping.

    Malformed containers (not a list, entries not tables) are left for
    CaseConfig's own Pydantic validation to reject with its normal error
    messages — matches _fill_missing_dataset_ids' convention of no-op on
    unexpected shape rather than a second, competing error path.
    """
    sweeps = raw.pop("dataset_sweeps", None)
    if not isinstance(sweeps, list) or not sweeps:
        return {}

    datasets = raw.setdefault("datasets", [])
    sweep_datasets: dict[str, list[dict[str, Any]]] = {}
    for sweep in sweeps:
        if not isinstance(sweep, dict):
            continue
        entries = _render_dataset_sweep_entries(sweep)
        label = sweep["label"]
        if label in sweep_datasets:
            raise ValueError(f"Duplicate [[dataset_sweeps]] label: '{label}'.")
        datasets.extend(entries)
        sweep_datasets[label] = entries
    return sweep_datasets


def _expand_assignment_sweeps(
    raw: dict[str, Any], sweep_datasets: dict[str, list[dict[str, Any]]]
) -> None:
    """Expand raw["assignment_sweeps"] into raw["assignments"] entries.

    Must run after _fill_missing_dataset_ids has resolved every dataset
    dict's 'id' in place (see _expand_dataset_sweeps' docstring) — this is
    where those resolved ids are read. Malformed containers are left for
    CaseConfig's own Pydantic validation, matching _expand_dataset_sweeps.
    """
    sweeps = raw.pop("assignment_sweeps", None)
    if not isinstance(sweeps, list) or not sweeps:
        return

    assignments = raw.setdefault("assignments", [])
    for sweep in sweeps:
        if not isinstance(sweep, dict):
            continue
        dataset_sweep = sweep.get("dataset_sweep")
        job_id = sweep.get("job")
        if not isinstance(dataset_sweep, str) or not dataset_sweep.strip():
            raise ValueError("[[assignment_sweeps]] entry is missing a non-blank 'dataset_sweep'.")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError(f"[[assignment_sweeps]] entry '{dataset_sweep}' is missing 'job'.")
        entries = sweep_datasets.get(dataset_sweep)
        if entries is None:
            raise ValueError(
                f"[[assignment_sweeps]] entry references unknown dataset_sweeps label "
                f"'{dataset_sweep}'."
            )
        for entry in entries:
            dataset_id = entry.get("id")
            if not isinstance(dataset_id, str) or not dataset_id.strip():
                raise ValueError(
                    f"Dataset sweep '{dataset_sweep}' entry has no resolved id (internal "
                    "ordering bug: assignment sweeps must expand after dataset ids are filled)."
                )
            assignments.append({"dataset": dataset_id, "job": job_id})


def load_case_config(path: Path, settings: NeurallsSettings) -> CaseConfig:
    """Load and validate the top-level case TOML."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    sweep_datasets = _expand_dataset_sweeps(raw, ctx)
    _fill_missing_dataset_ids(raw, ctx)
    _expand_assignment_sweeps(raw, sweep_datasets)
    return CaseConfig.model_validate(raw, context=ctx.as_pydantic_context())


def load_case(path: Path, env_file: Path | None = None) -> tuple[CaseConfig, NeurallsSettings]:
    """Load one case config together with its resolved runtime settings."""
    settings = load_case_settings(path, env_file)
    return load_case_config(path, settings), settings


def load_tracking_config(path: Path | None = None) -> SharedTrackingSettings | None:
    """Load the shared dlkit tracking config from configs/tracking.toml [tracking] section.

    Args:
        path: Optional explicit path. Defaults to ``configs/tracking.toml`` at project root.

    Returns:
        Validated ``SharedTrackingSettings`` when the file exists and has a [tracking] section,
        ``None`` if the file is missing or has no [tracking] section.
    """
    resolved = path or _DEFAULT_TRACKING_TOML
    if not resolved.exists():
        return None
    raw = load_raw_toml(resolved)
    section = raw.get("tracking")
    if not section:
        return None
    return SharedTrackingSettings.model_validate(section)
