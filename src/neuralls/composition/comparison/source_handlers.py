"""Source-specific comparison input handlers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.composition.comparison.rhs_generation import (
    generate_gaussian_rhs,
    generate_sinusoidal_rhs,
)
from neuralls.domain.masking import apply_mask, random_support_mask
from neuralls.domain.solver.models.provenance import MatrixProvenance
from neuralls.platform.config.models.comparison import (
    GaussianRhsSourceModel,
    NoMask,
    RandomMask,
    RawLhsSourceModel,
    RawRhsSourceModel,
    RawVectorSourceBase,
    RhsMask,
    SinusoidalRhsSourceModel,
)
from neuralls.platform.storage.comparison import (
    external_vector_length,
    load_system_arrays,
    matrix_file_order,
)
from neuralls.platform.storage.dataset_readers import (
    list_available_matrix_indices,
    resolve_canonical_training_triplet,
    try_read_dataset_normalization,
)
from neuralls.platform.storage.dataset_shape import system_dimension
from neuralls.platform.storage.validation import (
    validate_comparison_inputs,
    validate_comparison_matrix_input,
    validate_comparison_rhs_input,
)
from neuralls.shared.constants import MASK_STREAM_TAG
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
    matrix_index: int
    """Physical storage row of the matrix to solve on."""
    matrix_provenance: MatrixProvenance
    """What the planner proved about the matrix; stamped onto the resolved input."""
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


def require_stored_matrix_row(matrix_path: Path, matrix_index: int) -> int:
    """Return `matrix_index` after checking the dataset actually stores that matrix row."""
    if not matrix_path.is_dir():
        return matrix_index
    stored_rows = list_available_matrix_indices(matrix_path)
    if not stored_rows:
        raise ValueError(f"No matrix samples are available in comparison dataset '{matrix_path}'.")
    if matrix_index not in stored_rows:
        raise ValueError(
            f"matrix_index={matrix_index} is not valid for comparison dataset '{matrix_path}'."
        )
    return matrix_index


def _load_matrix(matrix_path: Path, matrix_index: int) -> np.ndarray:
    matrix, _dummy_rhs = load_system_arrays(
        matrix_path=matrix_path,
        rhs_path=matrix_path,
        rhs_sample_index=0,
        matrix_index=matrix_index,
    )
    return matrix


def _require_external_vector_fits(spec: ComparisonSourceSpec, source: RawVectorSourceBase) -> None:
    """Fail early when an external vector cannot be applied to the system's native matrix.

    A raw vector is applied to the dataset's own matrix, so its length must
    equal that matrix's order. Checking here fails only the one system, before
    any MLflow run is opened, instead of surfacing from link-time validation.
    """
    expected = (
        system_dimension(spec.matrix_path)
        if spec.matrix_path.is_dir()
        else matrix_file_order(spec.matrix_path)
    )
    try:
        found = external_vector_length(source.path, source.sample_index)
    except IndexError as exc:
        raise ValueError(f"External vector '{source.path}': {exc}") from exc
    if found != expected:
        raise ValueError(
            f"External vector '{source.path}' has length {found}, but the matrix of "
            f"'{spec.matrix_path}' has size {expected}."
        )


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
    """Resolve generated Gaussian or sinusoidal RHS sources."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """Only the matrix is read from disk — the RHS is synthesized."""
        validate_comparison_matrix_input(spec.matrix_path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = require_stored_matrix_row(ctx.matrix_path, ctx.matrix_index)
        matrix = _load_matrix(ctx.matrix_path, matrix_index)
        match ctx.rhs_source_kind:
            case ComparisonRhsSourceKind.GAUSSIAN:
                cfg = GaussianRhsSourceModel.model_validate(
                    {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
                )
                rhs = generate_gaussian_rhs(matrix.shape[0], cfg, ctx.seed)
            case ComparisonRhsSourceKind.SINUSOIDAL:
                cfg = SinusoidalRhsSourceModel.model_validate(
                    {"kind": ctx.rhs_source_kind, **(ctx.rhs_source_params or {})}
                )
                rhs = generate_sinusoidal_rhs(matrix.shape[0], cfg, ctx.seed)
            case _:
                raise ValueError(
                    f"Unsupported generated comparison RHS source: {ctx.rhs_source_kind}."
                )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=rhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=matrix_index,
            matrix_provenance=ctx.matrix_provenance,
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
        _require_external_vector_fits(spec, source)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = require_stored_matrix_row(ctx.matrix_path, ctx.matrix_index)
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
            matrix_provenance=ctx.matrix_provenance,
            rhs_sample_index=cfg.sample_index,
            row_kind=cfg.row_kind,
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
        _require_external_vector_fits(spec, source)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        matrix_index = require_stored_matrix_row(ctx.matrix_path, ctx.matrix_index)
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
            matrix_provenance=ctx.matrix_provenance,
            rhs_sample_index=cfg.sample_index,
            row_kind=cfg.row_kind,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=try_read_dataset_normalization(ctx.matrix_path),
        )


class RhsMaskParams(BaseModel):
    """The mask of any source's params, read without knowing the rest of them."""

    mask: RhsMask = NoMask()

    model_config = ConfigDict(extra="ignore", frozen=True)


class DatasetRowParams(RhsMaskParams):
    """The row of the entry's own dataset the planner bound to one `dataset`-kind system."""

    sample_index: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class DatasetSourceHandler:
    """Resolve canonical manifest-backed dataset triplets from the entry's own dataset."""

    def validate(self, spec: ComparisonSourceSpec) -> None:
        """The dataset supplies matrix and RHS together, so only its own path is checked."""
        validate_comparison_rhs_input(spec.matrix_path)

    def resolve(self, ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
        row = DatasetRowParams.model_validate(ctx.rhs_source_params or {})
        if not ctx.matrix_path.is_dir():
            raise ValueError(f"Dataset RHS source must be a directory, got file: {ctx.matrix_path}")
        triplet = resolve_canonical_training_triplet(
            ctx.matrix_path,
            row.sample_index,
            require_standard=ctx.require_non_residual_rhs,
            matrix_index=ctx.matrix_index,
        )
        return ResolvedComparisonInput(
            matrix=triplet.matrix,
            rhs=triplet.rhs,
            lhs=triplet.lhs,
            matrix_dataset_id=ctx.matrix_dataset_id,
            matrix_index=ctx.matrix_index,
            matrix_provenance=ctx.matrix_provenance,
            rhs_dataset_id=str(ctx.matrix_path),
            rhs_sample_index=triplet.sample_index,
            row_kind=triplet.row_kind,
            rhs_source_kind=ctx.rhs_source_kind,
            rhs_source_params=ctx.rhs_source_params,
            matrix_normalization=triplet.normalization,
        )


_SOURCE_HANDLERS: dict[ComparisonRhsSourceKind, ComparisonSourceHandler] = {
    ComparisonRhsSourceKind.GAUSSIAN: GeneratedSourceHandler(),
    ComparisonRhsSourceKind.SINUSOIDAL: GeneratedSourceHandler(),
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


def _apply_rhs_mask(
    resolved: ResolvedComparisonInput, mask: RhsMask, seed: int | None
) -> ResolvedComparisonInput:
    """Zero the RHS outside the mask's support.

    The mask has a random stream of its own, spawned from the system seed with
    `MASK_STREAM_TAG`, so it never shifts the RHS draw yet stays reproducible
    per system. A known solution no longer solves ``Ax = b`` once ``b`` is
    masked, so it is dropped rather than kept as a misleading reference.
    """
    match mask:
        case NoMask():
            return resolved
        case RandomMask(density=density):
            rng = np.random.default_rng(np.random.SeedSequence([seed or 0, MASK_STREAM_TAG]))
            support = random_support_mask(resolved.rhs.shape[0], density, rng)
            return replace(resolved, rhs=apply_mask(resolved.rhs, support), lhs=None)


def resolve_comparison_source(ctx: ComparisonSourceContext) -> ResolvedComparisonInput:
    """Dispatch source resolution, then mask the RHS, without workflow-level branching."""
    resolved = _require_handler(ctx.rhs_source_kind).resolve(ctx)
    mask = RhsMaskParams.model_validate(ctx.rhs_source_params or {}).mask
    return _apply_rhs_mask(resolved, mask, ctx.seed)
