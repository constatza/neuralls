"""Assembly of ComparisonConfig from case registry, defaults, and method overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from neuralls.domain.solver.models.config import (
    ComparisonData,
    ComparisonGeneral,
    SolverParams,
)
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.loaders import load_data_config, load_raw_toml
from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    parse_comparison_method_override,
)
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonDefaults,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.registry import resolve_dataset_config_path
from neuralls.platform.config.resolution import resolve_registry_path
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.types import ComparisonRhsSourceKind


def _resolve_runtime_index(
    entry: ComparisonRegistryEntry,
    field_name: Literal["matrix_index"],
) -> int | None:
    """Translate case-config defaults into runtime omission semantics."""
    if field_name not in entry.model_fields_set:
        return None
    return getattr(entry, field_name)


def resolve_comparison_config(
    case_cfg: CaseConfig,
    config_dir: Path,
    entry: ComparisonRegistryEntry,
    settings: NeurallsSettings,
) -> ComparisonConfig:
    """Build a fully resolved ComparisonConfig from a case comparison entry.

    Merges ``[comparison_defaults]`` with an optional methodology override file,
    then injects the data paths resolved from the dataset registry.

    Args:
        case_cfg: Top-level case configuration.
        config_dir: Directory of the case config file (for resolving relative paths).
        entry: The specific ``[[comparisons]]`` entry to resolve.
        settings: ``NeurallsSettings`` for processed-dir expansion.

    Returns:
        Fully materialised ``ComparisonConfig`` with injected paths.

    Raises:
        ValueError: If required defaults or datasets are missing.
    """
    defaults: ComparisonDefaults | None = case_cfg.comparison_defaults
    if defaults is None:
        raise ValueError(
            "Case config must define [comparison_defaults] to use the [[comparisons]] registry."
        )

    preconditioners: list[PreconditionerConfig] = list(defaults.preconditioners)
    normalize_system = defaults.normalize_system
    params_override: dict[str, Any] | None = None

    if entry.method is not None:
        method_path = resolve_registry_path(config_dir, entry.method)
        ctx = ConfigContext(config_path=method_path.resolve(), settings=settings)
        raw = load_raw_toml(method_path)
        override = parse_comparison_method_override(raw, ctx)
        if override.preconditioners:
            preconditioners = list(override.preconditioners)
        if "normalize_system" in override.general.data.model_fields_set:
            normalize_system = override.general.data.normalize_system
        if override.general.params is not None:
            params_override = override.general.params.model_dump()

    matrix_data_cfg = load_data_config(
        resolve_dataset_config_path(case_cfg, config_dir, entry.matrix_dataset),
        settings,
    )
    matrix_path = (
        Path(matrix_data_cfg.output.data_dir) / matrix_data_cfg.id
        if matrix_data_cfg.output.data_dir is not None
        else settings.processed_dir / matrix_data_cfg.id
    )
    base_params: dict[str, Any] = {
        "rtol": defaults.rtol,
        "atol": defaults.atol,
        "max_iterations": defaults.max_iterations,
        "stopping_criterion": defaults.stopping_criterion,
        "m_max": defaults.m_max,
    }
    if params_override:
        base_params.update(params_override)

    return ComparisonConfig(
        general=ComparisonGeneral(
            params=SolverParams(**base_params),
            data=ComparisonData(
                matrix_path=matrix_path,
                rhs_path=None,
                matrix_index=_resolve_runtime_index(entry, "matrix_index"),
                normalize_system=normalize_system,
                require_non_residual_rhs=entry.require_non_residual_rhs,
                selection_seed=entry.seed,
                rhs_source_kind=ComparisonRhsSourceKind(entry.rhs_source.kind),
                rhs_source_params=entry.rhs_source.model_dump(exclude={"kind"}),
            ),
        ),
        preconditioners=tuple(preconditioners),
    )
