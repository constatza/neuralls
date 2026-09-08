"""Source-specific comparison input handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.composition.comparison.rhs_generation import (
    generate_gaussian_rhs,
    generate_sparse_rhs,
)
from neuralls.platform.config.models.comparison import (
    DatasetRhsSourceModel,
    GaussianRhsSourceModel,
    RawLhsSourceModel,
    RawRhsSourceModel,
    SparseRhsSourceModel,
)
from neuralls.platform.storage.comparison import load_system_arrays
from neuralls.platform.storage.dataset_readers import (
    list_available_matrix_indices,
    resolve_canonical_training_triplet,
    try_read_dataset_normalization,
)
from neuralls.platform.storage.validation import (
    validate_comparison_inputs,
    validate_comparison_matrix_input,
    validate_comparison_rhs_input,
)
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


@dataclass(frozen=True)
class ComparisonSourceSpec:
    """A comparison's declared inputs, before any of them are read.

    The subset of `ComparisonSourceContext` that preflight validation needs:
    which files a source kind *says* it will read, with none of the sample
    selection that only matters once they are actually loaded.
    """

    matrix_path: Path
    rhs_source_kind: ComparisonRhsSourceKind
    rhs_source_params: dict[str, object] | None


@dataclass(frozen=True)
class ComparisonSourceContext:
    """Inputs shared by concrete RHS source handlers."""

    matrix_path: Path
    matrix_dataset_id: str
    matrix_index: int | None
    require_non_residual_rhs: bool
    seed: int | None
    rhs_source_kind: ComparisonRhsSourceKind
    rhs_source_params: dict[str, object] | None


class ComparisonSourceHandler(Protocol):
    """Resolve one concrete comparison RHS source kind.

    ``validate`` is the cheap preflight half of ``resolve``: it checks only
    the concrete input artifacts this kind will actually read, without loading
    any of them, so a missing input fails before any MLflow run is opened. The
    two halves live on the same handler so a new source kind cannot be added
    to resolution while a second, parallel validation dispatch silently keeps
    rejecting it.
    """

    def validate(self, spec: ComparisonSourceSpec) -> None: ...

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput: ...


def resolve_source_matrix_index(
    matrix_path: Path, matrix_index: int | None, seed: int | None
) -> int:
    """Resolve the matrix index used by generated/raw sources."""
    if not matrix_path.is_dir():
        return matrix_index if matrix_index is not None else 0
    matrix_indices = list_available_matrix_indices(matrix_path)
    if not matrix_indices:
        raise ValueError(f"No matrix samples are available in comparison dataset '{matrix_path}'.")
    resolved = matrix_index
    if resolved is None:
        rng = np.random.default_rng(seed)
        resolved = int(rng.choice(np.asarray(matrix_indices)))
    if resolved not in matrix_indices:
        raise ValueError(
            f"matrix_index={resolved} is not valid for comparison dataset '{matrix_path}'."
        )
    return resolved


def _load_matrix(matrix_path: Path, matrix_index: int) -> np.ndarray:
    matrix, _dummy_rhs = load_system_arrays(
        matrix_path=matrix_path,
        rhs_path=matrix_path,
        rhs_sample_index=0,
        matrix_index=matrix_index,
    )
    return matrix


def _validate_raw_row_kind(
    row_kind: RowKind | None,
    *,
    require_non_residual_rhs: bool,
    source_path: Path,
) -> None:
    if not require_non_residual_rhs:
        return
    if row_kind is None:
        raise ValueError(
            f"Raw comparison source '{source_path}' must define row_kind when "
            "non-residual RHS enforcement is enabled."
        )
    if row_kind is RowKind.CG_INTERNAL:
        raise ValueError(
            f"Raw comparison source '{source_path}' is marked as CG_INTERNAL and cannot be used "
            "with non-residual RHS enforcement."
        )


@dataclass(frozen=True)
class GeneratedSourceHandler:
    """Resolve generated Gaussian or sparse RHS sources."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """Only the matrix is read from disk — the RHS is synthesized."""
        validate_comparison_matrix_input(spec.matrix_path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = resolve_source_matrix_index(ctx.matrix_path, ctx.matrix_index, ctx.seed)
        matrix = _load_matrix(ctx.matrix_path, matrix_index)
        match ctx.rhs_source_kind:
            case ComparisonRhsSourceKind.GAUSSIAN:
                cfg = GaussianRhsSourceModel.model_validate(
                    {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
                )
                rhs = generate_gaussian_rhs(matrix.shape[0], cfg, ctx.seed)
            case ComparisonRhsSourceKind.SPARSE:
                cfg = SparseRhsSourceModel.model_validate(
                    {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
                )
                rhs = generate_sparse_rhs(matrix.shape[0], cfg)
            case _:
                raise ValueError(
                    f"Unsupported generated comparison RHS source: {ctx.rhs_source_kind}."
                )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=rhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=matrix_index,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=try_read_dataset_normalization(ctx.matrix_path),
        )


@dataclass(frozen=True)
class RawLhsSourceHandler:
    """Resolve raw LHS vectors by computing `b = scale * A @ x`."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """Both the matrix and the external LHS vector are read from disk."""
        source = RawLhsSourceModel.model_validate(
            {"kind": spec.rhs_source_kind, **(spec.rhs_source_params or {})}
        )
        validate_comparison_inputs(matrix_path=spec.matrix_path, rhs_path=source.path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = resolve_source_matrix_index(ctx.matrix_path, ctx.matrix_index, ctx.seed)
        cfg = RawLhsSourceModel.model_validate(
            {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
        )
        if cfg.path.is_dir():
            raise ValueError(f"Raw LHS source must be a file, got directory: {cfg.path}")
        _validate_raw_row_kind(
            cfg.row_kind,
            require_non_residual_rhs=ctx.require_non_residual_rhs,
            source_path=cfg.path,
        )
        matrix, lhs = load_system_arrays(
            matrix_path=ctx.matrix_path,
            rhs_path=cfg.path,
            rhs_sample_index=cfg.sample_index,
            matrix_index=matrix_index,
        )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=float(cfg.scale) * matrix @ lhs,
            lhs=lhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=matrix_index,
            rhs_sample_index=cfg.sample_index,
            rhs_kind=cfg.row_kind,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=try_read_dataset_normalization(ctx.matrix_path),
        )


@dataclass(frozen=True)
class RawRhsSourceHandler:
    """Resolve raw RHS vectors directly as `b`."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """Both the matrix and the external RHS vector are read from disk."""
        source = RawRhsSourceModel.model_validate(
            {"kind": spec.rhs_source_kind, **(spec.rhs_source_params or {})}
        )
        validate_comparison_inputs(matrix_path=spec.matrix_path, rhs_path=source.path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = resolve_source_matrix_index(ctx.matrix_path, ctx.matrix_index, ctx.seed)
        cfg = RawRhsSourceModel.model_validate(
            {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
        )
        if cfg.path.is_dir():
            raise ValueError(f"Raw RHS source must be a file, got directory: {cfg.path}")
        _validate_raw_row_kind(
            cfg.row_kind,
            require_non_residual_rhs=ctx.require_non_residual_rhs,
            source_path=cfg.path,
        )
        matrix, rhs = load_system_arrays(
            matrix_path=ctx.matrix_path,
            rhs_path=cfg.path,
            rhs_sample_index=cfg.sample_index,
            matrix_index=matrix_index,
        )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=float(cfg.scale) * rhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=matrix_index,
            rhs_sample_index=cfg.sample_index,
            rhs_kind=cfg.row_kind,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=try_read_dataset_normalization(ctx.matrix_path),
        )


@dataclass(frozen=True)
class DatasetSourceHandler:
    """Resolve canonical manifest-backed dataset triplets."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """The dataset supplies matrix and RHS together, so only its own path is checked."""
        source = DatasetRhsSourceModel.model_validate(
            {"kind": spec.rhs_source_kind, **(spec.rhs_source_params or {})}
        )
        validate_comparison_rhs_input(source.path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        cfg = DatasetRhsSourceModel.model_validate(
            {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
        )
        if not cfg.path.is_dir():
            raise ValueError(f"Dataset RHS source must be a directory, got file: {cfg.path}")
        triplet = resolve_canonical_training_triplet(
            cfg.path,
            cfg.sample_index,
            require_standard=ctx.require_non_residual_rhs,
            matrix_index=ctx.matrix_index,
        )
        return ResolvedComparisonInput(
            matrix=triplet.matrix,
            rhs=triplet.rhs,
            lhs=triplet.lhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=triplet.matrix_index,
            rhs_dataset_id=str(cfg.path),
            rhs_sample_index=triplet.sample_index,
            rhs_kind=triplet.row_kind,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=triplet.normalization,
        )


_SOURCE_HANDLERS: dict[ComparisonRhsSourceKind, ComparisonSourceHandler] = {
    ComparisonRhsSourceKind.GAUSSIAN: GeneratedSourceHandler(),
    ComparisonRhsSourceKind.SPARSE: GeneratedSourceHandler(),
    ComparisonRhsSourceKind.RAW_LHS: RawLhsSourceHandler(),
    ComparisonRhsSourceKind.RAW_RHS: RawRhsSourceHandler(),
    ComparisonRhsSourceKind.DATASET: DatasetSourceHandler(),
}


def _require_handler(kind: ComparisonRhsSourceKind) -> ComparisonSourceHandler:
    """Look up the handler for one RHS source kind.

    Raises:
        ValueError: If no handler is registered for the kind.
    """
    handler = _SOURCE_HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"Unsupported comparison RHS source: {kind!r}.")
    return handler


def validate_comparison_source(spec: ComparisonSourceSpec) -> None:
    """Prevalidate one source's concrete inputs without workflow-level branching."""
    _require_handler(spec.rhs_source_kind).validate(spec)


def resolve_comparison_source(ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
    """Dispatch source resolution without workflow-level branching."""
    return _require_handler(ctx.rhs_source_kind).resolve(ctx)
