"""Assembly of ComparisonConfig from case registry, defaults, and method overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuralls.composition.comparison.models import PlannedSystem
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
    ComparisonBaseline,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.registry import resolve_dataset_config_path
from neuralls.platform.config.resolution import resolve_registry_path
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.types import ComparisonRhsSourceKind, PreconditionerName


def resolve_dataset_dir(
    case_cfg: CaseConfig,
    config_dir: Path,
    dataset_id: str,
    settings: NeurallsSettings,
) -> Path:
    """Resolve a ``[[datasets]]`` registry id into its concrete dataset directory.

    Shared by ``resolve_comparison_config`` and the comparison planner so both
    read the same dataset directory the same way, instead of re-deriving it.
    """
    data_cfg = load_data_config(
        resolve_dataset_config_path(case_cfg, config_dir, dataset_id), settings
    )
    return (
        Path(data_cfg.output.data_dir) / data_cfg.id
        if data_cfg.output.data_dir is not None
        else settings.processed_dir / data_cfg.id
    )


def _rhs_source_params(planned: PlannedSystem) -> dict[str, object]:
    """The source's own params, plus the held-out row the planner bound for a `dataset` source."""
    params = planned.entry.rhs_source.model_dump(exclude={"kind"})
    if planned.rhs_row is not None:
        params["sample_index"] = planned.rhs_row
    return params


def resolve_comparison_config(
    case_cfg: CaseConfig,
    config_dir: Path,
    planned: PlannedSystem,
    settings: NeurallsSettings,
) -> ComparisonConfig:
    """Build a fully resolved ComparisonConfig from one planned system.

    Merges ``[comparison_baseline]`` with an optional methodology override file,
    then injects the data paths resolved from the dataset registry and the
    matrix row, RHS seed and RHS row the planner bound to this system.

    Args:
        case_cfg: Top-level case configuration.
        config_dir: Directory of the case config file (for resolving relative paths).
        planned: The planned system to resolve.
        settings: ``NeurallsSettings`` for processed-dir expansion.

    Returns:
        Fully materialised ``ComparisonConfig`` with injected paths.

    Raises:
        ValueError: If the case comparison baseline or a referenced dataset is missing.
    """
    entry = planned.entry
    baseline: ComparisonBaseline | None = case_cfg.comparison_baseline
    if baseline is None:
        raise ValueError(
            "Case config must define [comparison_baseline] to use the [[comparisons]] registry."
        )

    preconditioners: list[PreconditionerConfig] = list(baseline.preconditioners)
    reference_preconditioner = baseline.reference_preconditioner
    normalize_system = baseline.normalize_system
    presentation = baseline.presentation
    params_override: dict[str, Any] | None = None

    if entry.method is not None:
        method_path = resolve_registry_path(config_dir, entry.method)
        ctx = ConfigContext(config_path=method_path.resolve(), settings=settings)
        raw = load_raw_toml(method_path)
        override = parse_comparison_method_override(raw, ctx)
        if override.preconditioners:
            preconditioners = list(override.preconditioners)
        if "reference_preconditioner" in override.general.model_fields_set:
            reference_preconditioner = override.general.reference_preconditioner
        if "normalize_system" in override.general.data.model_fields_set:
            normalize_system = override.general.data.normalize_system
        if override.general.params is not None:
            params_override = dict(override.general.params)
        if override.presentation is not None:
            presentation = override.presentation

    matrix_path = resolve_dataset_dir(case_cfg, config_dir, entry.matrix_dataset, settings)
    base_params: dict[str, Any] = {
        "rtol": baseline.rtol,
        "atol": baseline.atol,
        "max_iterations": baseline.max_iterations,
        "stopping_criterion": baseline.stopping_criterion,
        "m_max": baseline.m_max,
        "timing": baseline.timing,
    }
    if params_override:
        base_params.update(params_override)

    return ComparisonConfig(
        general=ComparisonGeneral(
            params=SolverParams(**base_params),
            data=ComparisonData(
                matrix_path=matrix_path,
                rhs_path=None,
                matrix_index=planned.matrix.row,
                normalize_system=normalize_system,
                require_non_residual_rhs=entry.require_non_residual_rhs,
                selection_seed=entry.seed,
                rhs_source_kind=ComparisonRhsSourceKind(entry.rhs_source.kind),
                rhs_source_params=_rhs_source_params(planned),
            ),
            reference_preconditioner=(
                None
                if reference_preconditioner is None
                else PreconditionerName(reference_preconditioner)
            ),
        ),
        preconditioners=tuple(preconditioners),
        notation_policy=presentation.notation_policy(),
    )
