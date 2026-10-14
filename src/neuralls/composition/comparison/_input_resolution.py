"""Resolve one comparison input system from a canonical RHS source."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.composition.comparison.source_handlers import (
    ComparisonSourceContext,
    resolve_comparison_source,
)
from neuralls.domain.solver.models.provenance import MatrixProvenance
from neuralls.shared.types import ComparisonRhsSourceKind


def _normalize_seed(seed: object) -> int | None:
    """Coerce optional seed-like values coming from config or test doubles."""
    return seed if isinstance(seed, int) else None


def _normalize_rhs_source_kind(kind: object) -> ComparisonRhsSourceKind:
    """Coerce source-kind values coming from config or test doubles."""
    if isinstance(kind, ComparisonRhsSourceKind):
        return kind
    if isinstance(kind, str):
        return ComparisonRhsSourceKind(kind)
    raise ValueError(f"Unsupported comparison RHS source: {kind!r}.")


def resolve_comparison_input(
    *,
    matrix_path: Path,
    matrix_dataset_id: str,
    matrix_index: int,
    matrix_provenance: MatrixProvenance,
    require_non_residual_rhs: bool,
    seed: int | None,
    rhs_source_kind: ComparisonRhsSourceKind | str,
    rhs_source_params: dict[str, object] | None,
) -> ResolvedComparisonInput:
    """Resolve the final comparison system and provenance before solver execution."""
    ctx = ComparisonSourceContext(
        matrix_path=matrix_path,
        matrix_dataset_id=matrix_dataset_id,
        matrix_index=matrix_index,
        matrix_provenance=matrix_provenance,
        require_non_residual_rhs=require_non_residual_rhs,
        seed=_normalize_seed(seed),
        rhs_source_kind=_normalize_rhs_source_kind(rhs_source_kind),
        rhs_source_params=rhs_source_params,
    )
    return resolve_comparison_source(ctx)
