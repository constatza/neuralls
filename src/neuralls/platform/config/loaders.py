"""TOML loading and validation for neuralls-owned config types."""

from __future__ import annotations

import itertools
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


_AXIS_SCALAR_TYPES = (str, int, float, bool)


def _combine_axes(axes: dict[str, Any], *, cart_product: bool) -> list[dict[str, Any]]:
    """One row per axis combination; scalar axes broadcast, list axes combine.

    List-valued axes combine via itertools.product (cart_product=True) or
    zip(..., strict=True) (cart_product=False, default — equal length
    required, matching this codebase's existing strict-zip convention, e.g.
    domain/generation/rhs_generation.py's indices/values pairing). A scalar
    axis has nothing to combine against, so it's merged into every resulting
    row unchanged, regardless of mode — this is what makes "any axis may be
    a scalar or a list" a single rule rather than a special case per axis.

    Assumes axes has already been shape-validated (see _validate_axis_values)
    — this function's only job is combining, not validating.
    """
    list_axes = {name: value for name, value in axes.items() if isinstance(value, list)}
    scalar_axes = {name: value for name, value in axes.items() if not isinstance(value, list)}
    if not list_axes:
        raise ValueError("[[dataset_sweeps]] entry has no list-valued axis to sweep.")

    names = list(list_axes.keys())
    combos = (
        itertools.product(*(list_axes[name] for name in names))
        if cart_product
        else zip(*(list_axes[name] for name in names), strict=True)
    )
    return [scalar_axes | dict(zip(names, combo, strict=True)) for combo in combos]


def _substitute_template(template: str, row: dict[str, Any]) -> str:
    """Replace every {axis_name} placeholder in template with its row value.

    Uses str.replace per axis rather than str.format so an unrelated
    ${...}-shaped substring elsewhere in the template (none exist in
    practice, but path templates are free-form strings) can never be
    misread as a format field.
    """
    result = template
    for name, value in row.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def _require_dataset_sweep_label(sweep: dict[str, Any]) -> str:
    """Extract and validate one [[dataset_sweeps]] entry's 'label'."""
    label = sweep.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("[[dataset_sweeps]] entry is missing a non-blank 'label'.")
    return label


def _require_dataset_sweep_path_template(sweep: dict[str, Any], label: str) -> str:
    """Extract and validate one [[dataset_sweeps]] entry's 'path_template'."""
    path_template = sweep.get("path_template")
    if not isinstance(path_template, str) or not path_template:
        raise ValueError(f"[[dataset_sweeps]] entry '{label}' has no 'path_template'.")
    return path_template


def _validate_axis_values(axes: dict[str, Any], label: str) -> None:
    """Reject a nested table (or an empty list) as an axis value.

    An axis must be a plain scalar (broadcast) or a non-empty list of plain
    scalars (combined) — never a table, and never empty. Without this, a
    stray nested table would silently stringify into a garbage path segment
    instead of failing at the point of the mistake.
    """
    for name, value in axes.items():
        candidates = value if isinstance(value, list) else [value]
        if not candidates or any(not isinstance(item, _AXIS_SCALAR_TYPES) for item in candidates):
            raise ValueError(
                f"[[dataset_sweeps]] entry '{label}' axis '{name}' must be a scalar or a "
                "non-empty list of scalars (no nested tables/lists)."
            )


def _resolve_dataset_sweep_axes(sweep: dict[str, Any], label: str) -> dict[str, Any]:
    """Resolve one [[dataset_sweeps]] entry's axes, from either 'axes' or legacy 'values'."""
    axes = sweep.get("axes")
    values = sweep.get("values")
    if axes is not None and values is not None:
        raise ValueError(
            f"[[dataset_sweeps]] entry '{label}' has both 'axes' and 'values' — use one."
        )
    if axes is not None:
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"[[dataset_sweeps]] entry '{label}' has an empty or invalid 'axes'.")
        resolved = axes
    else:
        if not isinstance(values, list) or not values:
            raise ValueError(f"[[dataset_sweeps]] entry '{label}' has no non-empty 'values' list.")
        resolved = {"value": values}
    _validate_axis_values(resolved, label)
    return resolved


def _validate_template_covers_axes(path_template: str, axes: dict[str, Any], label: str) -> None:
    """Reject an axis nobody's path_template references — almost certainly a typo."""
    for axis_name in axes:
        if f"{{{axis_name}}}" not in path_template:
            raise ValueError(
                f"[[dataset_sweeps]] entry '{label}' declares axis '{axis_name}' but "
                f"'path_template' has no '{{{axis_name}}}' placeholder for it."
            )


def _render_dataset_sweep_entries(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one [[dataset_sweeps]] entry into [[datasets]]-shaped dicts (id left unset).

    Pure — no I/O, no context. Each returned dict has only 'path' (and
    'display_name' when a template is given); 'id' is deliberately absent so
    the caller's subsequent _fill_missing_dataset_ids pass resolves it from
    the real dataset config it points at, the same way a hand-written
    [[datasets]] entry without an id would be resolved.

    Thin orchestrator: each validation/extraction step below is a single-
    purpose helper (label, path_template, axes, template-coverage), and the
    per-row work is delegated to _combine_axes (combining) and
    _substitute_template (rendering) — this function's only job is wiring
    them together in order.
    """
    label = _require_dataset_sweep_label(sweep)
    path_template = _require_dataset_sweep_path_template(sweep, label)
    axes = _resolve_dataset_sweep_axes(sweep, label)
    _validate_template_covers_axes(path_template, axes, label)
    display_name_template = sweep.get("display_name_template")
    cart_product = bool(sweep.get("cart_product", False))

    entries: list[dict[str, Any]] = []
    for row in _combine_axes(axes, cart_product=cart_product):
        entry: dict[str, Any] = {"path": _substitute_template(path_template, row)}
        if isinstance(display_name_template, str):
            entry["display_name"] = _substitute_template(display_name_template, row)
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


def _require_assignment_sweep_jobs(sweep: dict[str, Any], dataset_sweep: str) -> list[str]:
    """Extract and validate one [[assignment_sweeps]] entry's 'job' as a list of job ids.

    'job' may be a single string (that one job, broadcast to every dataset in
    the referenced dataset_sweep — today's original behavior) or a list of
    strings (every job crossed with every dataset in the referenced
    dataset_sweep, e.g. testing every POD-2G rank against every cg-variant
    instead of pairing them 1:1 by name).
    """
    job = sweep.get("job")
    jobs = job if isinstance(job, list) else [job] if isinstance(job, str) else None
    if not jobs or any(not isinstance(j, str) or not j.strip() for j in jobs):
        raise ValueError(
            f"[[assignment_sweeps]] entry '{dataset_sweep}' has no non-blank 'job' "
            "(a single job id string, or a non-empty list of job id strings)."
        )
    return jobs


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
        if not isinstance(dataset_sweep, str) or not dataset_sweep.strip():
            raise ValueError("[[assignment_sweeps]] entry is missing a non-blank 'dataset_sweep'.")
        job_ids = _require_assignment_sweep_jobs(sweep, dataset_sweep)
        entries = sweep_datasets.get(dataset_sweep)
        if entries is None:
            raise ValueError(
                f"[[assignment_sweeps]] entry references unknown dataset_sweeps label "
                f"'{dataset_sweep}'."
            )
        for job_id in job_ids:
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
