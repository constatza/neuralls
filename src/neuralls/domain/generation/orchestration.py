"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
from loguru import logger

from neuralls.domain.normalization import ErrorTraceSamples, IScale, ResidualTraceSamples
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import GenerationStrategyKind, LayoutType, RowKind, ScaleMetadata

from .helpers import (
    _merge_strategy_outputs,
    _resolve_strategy_counts,
    rng_from_seed,
    select_archive_files,
    serialize_scale_metadata,
)
from .interfaces import ArchiveData, TracingSolverCallable
from .payloads import GeneratedDatasetPayload
from .ports import DenseAccumulatorPort
from .runner import strategy_supports_matrix_replacement
from .semantics import classify_strategy_row_kind
from .source_streams import (
    MatrixSampleStream,
    SystemBinding,
    VectorSampleStream,
    bind_sources,
    open_matrix_stream,
    open_vector_stream,
)
from .specs import DatasetSpec, MixtureSpec, SourceSpec


@dataclass(frozen=True)
class _StrategyProperties:
    """Compile-time properties for a known generation strategy."""

    uses_finite_source: bool
    supports_replacement: bool


_ALL_SAMPLES: int = -1
"""Sentinel used in strategy count allocation: emit all available samples for this binding."""

_NORM_AGREEMENT_RTOL: float = 1e-10
_NORM_AGREEMENT_ATOL: float = 1e-12

_STRATEGY_PROPERTIES: dict[str, _StrategyProperties] = {
    "solution_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "rhs_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "scaled_solutions": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "validated_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "residuals": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
    "gaussian_residuals": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
    "search_directions": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
}


@dataclass(frozen=True)
class _GeneratedMixtureWithMetadata:
    """Private metadata-rich generation result used by dataset persistence."""

    rhs: np.ndarray
    solutions: np.ndarray
    residual_traces: ResidualTraceSamples | None
    error_traces: ErrorTraceSamples | None
    row_kind_codes: np.ndarray


def _reindex_residual_traces(
    traces: ResidualTraceSamples, inverse: np.ndarray
) -> ResidualTraceSamples:
    """Repoint residual traces at their base samples' new positions.

    Args:
        traces: Residual traces whose ``sample_indices`` address pre-shuffle rows.
        inverse: Inverse permutation mapping old row position to new.

    Returns:
        Traces with remapped ``sample_indices``; per-iteration arrays are shared as-is.
    """
    return ResidualTraceSamples(
        residuals=traces.residuals,
        solutions=traces.solutions,
        sample_indices=inverse[traces.sample_indices],
        iteration_indices=traces.iteration_indices,
        search_directions=traces.search_directions,
        search_direction_products=traces.search_direction_products,
    )


def _reindex_error_traces(
    traces: ErrorTraceSamples, indices: np.ndarray, inverse: np.ndarray
) -> ErrorTraceSamples:
    """Repoint error traces at their base samples' new positions.

    Args:
        traces: Error traces whose ``sample_indices`` address pre-shuffle rows.
        indices: Forward permutation, applied to the per-sample ``true_solutions``.
        inverse: Inverse permutation mapping old row position to new.

    Returns:
        Traces with permuted ``true_solutions`` and remapped ``sample_indices``.
    """
    return ErrorTraceSamples(
        residuals=traces.residuals,
        solutions_current=traces.solutions_current,
        errors=traces.errors,
        true_solutions=traces.true_solutions[indices],
        sample_indices=inverse[traces.sample_indices],
        iteration_indices=traces.iteration_indices,
    )


def _shuffle_samples(
    X: np.ndarray,
    Y: np.ndarray,
    residual_traces: ResidualTraceSamples | None,
    error_traces: ErrorTraceSamples | None,
    row_kind_codes: np.ndarray,
    indices: np.ndarray,
) -> _GeneratedMixtureWithMetadata:
    """Apply a permutation to samples and every array aligned with them.

    Pure: the caller draws the permutation, so the shuffle itself carries no
    RNG dependency. Per-sample arrays (features, targets, row kinds, trace
    ``true_solutions``) are permuted directly; trace rows are per-iteration
    rather than per-sample, so they are left in place and only their
    ``sample_indices`` are remapped through the inverse permutation.

    Args:
        X: Feature array, shape (N, n).
        Y: Target array, shape (N, n).
        residual_traces: Optional residual trace samples.
        error_traces: Optional error trace samples.
        row_kind_codes: Per-sample row-kind codes, shape (N,).
        indices: Permutation of ``range(N)`` to apply.

    Returns:
        The permuted samples, traces and row kinds as one metadata-rich result.
    """
    inverse = np.empty_like(indices)
    inverse[indices] = np.arange(len(indices))

    return _GeneratedMixtureWithMetadata(
        rhs=X[indices],
        solutions=Y[indices],
        residual_traces=(
            None if residual_traces is None else _reindex_residual_traces(residual_traces, inverse)
        ),
        error_traces=(
            None if error_traces is None else _reindex_error_traces(error_traces, indices, inverse)
        ),
        row_kind_codes=row_kind_codes[indices],
    )


def _generate_mixture_with_metadata(
    A: np.ndarray,
    mixture: MixtureSpec,
    *,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
    single_solution: np.ndarray | None = None,
) -> _GeneratedMixtureWithMetadata:
    """Generate mixed training data from multiple strategies.

    Args:
        A: System matrix, shape (n, n)
        mixture: Strategy counts/proportions and RNG controls for this call.
        archive_solutions: Pre-computed solutions for archive-based generation
        archive_rhs: Pre-computed RHS vectors for archive-based generation
        single_rhs: Optional single RHS vector, shape (n,). If provided to single-RHS strategies
            (trace strategies), all samples will solve the same system A @ x = single_rhs
        single_solution: Optional single pre-computed solution vector, shape (n,). If provided
            and no archive_solutions are present, it is used as a single-row archive for
            archive-based strategies.

    Returns:
        Metadata-rich generation result for dataset persistence internals.

    Raises:
        ValueError: If counts/mix arguments invalid or strategies unknown

    Examples:
        >>> # Generate 100 samples with equal mix of normal and krylov
        >>> X, Y, res_traces, err_traces = generate_mixture(
        ...     A,
        ...     mix={"normal": 1.0, "krylov": 1.0},
        ...     total=100,
        ...     seed=42,
        ...     strategy_overrides={
        ...         "krylov": {"krylov_iters": 20},
        ...     },
        ... )
        >>> X.shape
        (100, n)

        >>> # Generate explicit counts with strategy-specific configuration
        >>> X, Y, _, _ = generate_mixture(
        ...     A,
        ...     counts={"gaussian_forward": 50, "krylov": 30, "gaussian_residuals": 20},
        ...     seed=42,
        ...     strategy_overrides={
        ...         "gaussian_residuals": {"cg_iters": 10},
        ...     },
        ... )

        >>> # Generate with single RHS for trace strategies
        >>> rhs = np.random.randn(n)
        >>> X, Y, _, _ = generate_mixture(
        ...     A,
        ...     counts={"gaussian_residuals": 20},
        ...     single_rhs=rhs,  # All 20 samples solve A @ x = rhs
        ...     seed=42,
        ... )
    """
    # Ensure strategy modules are registered
    from pydantic import ValidationError

    from . import strategies  # noqa: F401
    from .runner import run_generation

    seed = mixture.seed
    solver_overrides = mixture.solver_overrides
    rng = rng_from_seed(seed)
    strategy_counts = _resolve_strategy_counts(mixture.counts, mixture.mix, mixture.total)
    overrides: dict[str, dict[str, Any]] = {
        name: dict(options) for name, options in (mixture.strategy_overrides or {}).items()
    }

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    row_kind_blocks: list[np.ndarray] = []

    for strategy_name, count in strategy_counts.items():
        if count == 0:
            continue

        cfg = overrides.get(strategy_name, {}).copy()
        cfg.setdefault("samples", count)
        cfg.setdefault("seed", seed)

        effective_archive_solutions = archive_solutions
        if single_solution is not None and archive_solutions is None:
            effective_archive_solutions = single_solution.reshape(1, -1)

        archive_data: ArchiveData | None = None
        if effective_archive_solutions is not None:
            archive_data = ArchiveData(lhs=effective_archive_solutions, rhs=archive_rhs)

        cfg["samples"] = count

        # Run generation (all strategies now use unified interface)
        # Single-RHS strategies (trace strategies) will receive single_rhs if provided
        # Pydantic (extra="forbid") will raise ValidationError on unknown keys — fail fast.
        try:
            generated = run_generation(
                strategy_name,
                A,
                cfg=cfg,
                solver=solver_overrides.get(strategy_name) if solver_overrides else None,
                archive=archive_data,
                single_rhs=single_rhs,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid configuration for strategy '{strategy_name}': {e}") from e

        # Trace strategies expose their training pairs via trace structs, not rhs/solutions.
        # Flatten them directly into the accumulated arrays so mixing strategies is correct.
        if generated.error_traces is not None:
            all_features.append(generated.error_traces.residuals)  # r_k
            all_targets.append(generated.error_traces.errors)  # e_k = x_true - x_k
        elif generated.residual_traces is not None:
            all_features.append(generated.residual_traces.residuals)  # A @ p_k
            all_targets.append(generated.residual_traces.solutions)  # p_k
        else:
            if (generated.rhs is None) != (generated.solutions is None):
                raise ValueError(
                    f"Strategy '{strategy_name}' returned rhs and solutions with inconsistent "
                    f"None-ness: rhs={'None' if generated.rhs is None else 'array'}, "
                    f"solutions={'None' if generated.solutions is None else 'array'}."
                )
            if generated.rhs is not None and generated.solutions is not None:
                all_features.append(generated.rhs)
                all_targets.append(generated.solutions)

        strategy_kind = GenerationStrategyKind(strategy_name)
        semantic_size = 0
        if generated.error_traces is not None:
            semantic_size = int(generated.error_traces.errors.shape[0])
        elif generated.residual_traces is not None:
            semantic_size = int(generated.residual_traces.residuals.shape[0])
        elif generated.rhs is not None:
            semantic_size = int(generated.rhs.shape[0])

        if semantic_size > 0:
            base_kind = classify_strategy_row_kind(strategy_kind)
            iter_indices = None
            if (et := generated.error_traces) is not None:
                iter_indices = et.iteration_indices
            elif (rt := generated.residual_traces) is not None:
                iter_indices = rt.iteration_indices
            # iter 0 is STANDARD only because CG initialises at x_0 = 0:
            # r_0 = b - A@0 = b  and  e_0 = x_true - 0 = x_true → (r_0, e_0) = (b, x_true).
            # WARNING: if the solver ever uses a non-zero initial guess this breaks silently.
            if iter_indices is not None:
                row_kinds = [RowKind.STANDARD if i == 0 else base_kind for i in iter_indices]
            else:
                row_kinds = [base_kind] * semantic_size
            row_kind_blocks.append(encode_row_kind_array(row_kinds))

    if all_features and all_targets:
        X, Y = _merge_strategy_outputs(all_features, all_targets)
    else:
        X = np.empty((0, A.shape[0]), dtype=np.float64)
        Y = np.empty((0, A.shape[0]), dtype=np.float64)
    row_kind_codes = (
        np.concatenate(row_kind_blocks) if row_kind_blocks else np.empty((0,), dtype=np.uint8)
    )

    if mixture.shuffle and X.shape[0] > 0:
        return _shuffle_samples(
            X,
            Y,
            None,
            None,
            row_kind_codes,
            rng.permutation(X.shape[0]),
        )

    return _GeneratedMixtureWithMetadata(
        rhs=X,
        solutions=Y,
        residual_traces=None,
        error_traces=None,
        row_kind_codes=row_kind_codes,
    )


def generate_mixture(
    A: np.ndarray,
    counts: Mapping[str, int] | None = None,
    *,
    mix: Mapping[str, float] | None = None,
    total: int | None = None,
    counts_represent_final_pairs: bool = False,
    seed: int = 42,
    shuffle: bool = True,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
    single_solution: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ResidualTraceSamples | None, ErrorTraceSamples | None]:
    """Generate mixed training data with the stable public 4-tuple API.

    Keeps the historical flat keyword signature; internally it assembles a
    :class:`MixtureSpec` and delegates. ``counts_represent_final_pairs`` is a
    no-op kept for call-site compatibility — trace strategies have interpreted
    counts as final output rows directly for some time.
    """
    result = _generate_mixture_with_metadata(
        A,
        MixtureSpec(
            counts=counts,
            mix=mix,
            total=total,
            seed=seed,
            shuffle=shuffle,
            strategy_overrides=strategy_overrides,
            solver_overrides=solver_overrides,
        ),
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
        single_rhs=single_rhs,
        single_solution=single_solution,
    )
    return result.rhs, result.solutions, result.residual_traces, result.error_traces


@dataclass(frozen=True)
class _CachedMatrix:
    """Immutable cached matrix data for generation.

    Stores all derived matrices and scaling information computed once
    per unique matrix_sample_id, avoiding redundant computation.

    Attributes:
        matrix_norm: Normalized system matrix (as dense array)
        scale: IScale object or None (scaling strategy applied)
        matrix_norm_value: Computed matrix norm value
        matrix_value_scale: Scaling factor applied
        scale_params: Dictionary of scale parameters or None
    """

    matrix_norm: np.ndarray
    scale: IScale | None
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


@dataclass(frozen=True)
class OpenedStreams:
    """Every source stream opened for one generation run, plus their bindings.

    Attributes:
        matrix: Opened matrix sample stream.
        rhs: Optional opened RHS vector stream.
        solution: Optional opened pre-computed solution vector stream.
        parameters: Opened parameter vector streams, in configured order.
        bindings: Resolved matrix/RHS/solution/parameter sample bindings.
    """

    matrix: MatrixSampleStream
    rhs: VectorSampleStream | None
    solution: VectorSampleStream | None
    parameters: tuple[VectorSampleStream, ...]
    bindings: tuple[SystemBinding, ...]

    @property
    def single_matrix_mode(self) -> bool:
        """Whether the matrix source holds exactly one matrix sample."""
        return len(self.matrix.sample_ids) == 1


def _open_streams(source: SourceSpec) -> OpenedStreams:
    """Open matrix, optional RHS, optional solution, and optional parameter streams and bind them.

    Args:
        source: Resolved source paths and per-stream sample filters.

    Returns:
        OpenedStreams holding every opened stream and the resolved bindings.
    """

    def _vector_stream(path_expr: str) -> VectorSampleStream:
        return open_vector_stream(
            path_expr,
            sample_id_regex=source.sample_id_regex,
            enumerate_by=source.enumerate_by,
            include_indices=source.include_indices,
            exclude_indices=source.exclude_indices,
        )

    matrix_stream = open_matrix_stream(
        matrix_path_expr=source.matrix_path,
        sample_id_regex=source.sample_id_regex,
        enumerate_by=source.enumerate_by,
        include_indices=source.include_indices,
        exclude_indices=source.exclude_indices,
    )
    rhs_stream = _vector_stream(source.rhs_path) if source.rhs_path is not None else None
    solution_stream = (
        _vector_stream(source.solution_path) if source.solution_path is not None else None
    )
    param_streams = tuple(_vector_stream(p) for p in source.parameters_paths)
    bindings = bind_sources(
        matrix_ids=matrix_stream.sample_ids,
        rhs_ids=rhs_stream.sample_ids if rhs_stream is not None else None,
        solution_ids=solution_stream.sample_ids if solution_stream is not None else None,
        parameters_ids_list=tuple(s.sample_ids for s in param_streams),
    )
    return OpenedStreams(
        matrix=matrix_stream,
        rhs=rhs_stream,
        solution=solution_stream,
        parameters=param_streams,
        bindings=tuple(bindings),
    )


def _strategy_uses_finite_source(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    has_rhs_source: bool,
) -> bool:
    """Return whether a strategy is configured to use a finite external source."""
    props = _STRATEGY_PROPERTIES.get(strategy_name)
    if props is not None:
        return props.uses_finite_source
    # Fallback heuristic for unknown/future strategies
    overrides = strategy_overrides.get(strategy_name, {}) if strategy_overrides is not None else {}
    return has_rhs_source or isinstance(overrides.get("solutions_glob"), str)


def _validate_replacement_support(
    strategy_counts: Mapping[str, int],
    *,
    replacement: bool,
    num_matrix_samples: int,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    has_rhs_source: bool,
) -> None:
    """Fail fast when replacement is requested or incompatible strategies are mixed."""
    if num_matrix_samples <= 1:
        return

    def _supports(name: str) -> bool:
        props = _STRATEGY_PROPERTIES.get(name)
        return (
            props.supports_replacement
            if props is not None
            else strategy_supports_matrix_replacement(name)
        )

    # count != 0 (not "> 0") so _ALL_SAMPLES (-1, "emit everything") is still
    # treated as active — otherwise a strategy requesting -1 would be invisible
    # to the mixing/replacement guards below, regardless of its actual output.
    active = [name for name, count in strategy_counts.items() if count != 0]
    multi_matrix = [name for name in active if _supports(name)]
    single_matrix = [name for name in active if not _supports(name)]

    if single_matrix and multi_matrix:
        raise ValueError(
            f"Cannot mix single-matrix strategies {single_matrix} with "
            f"multi-matrix strategies {multi_matrix}: "
            "the matrix source contains multiple matrices. Use a single-matrix source."
        )

    if not replacement:
        return

    if single_matrix:
        raise ValueError(
            f"Strategy '{single_matrix[0]}' does not support matrix replacement allocation."
        )
    for strategy_name in multi_matrix:
        if _strategy_uses_finite_source(strategy_name, strategy_overrides, has_rhs_source):
            raise ValueError(
                f"Strategy '{strategy_name}' does not support matrix replacement when configured "
                "with a finite external source."
            )


def _archive_glob_for_strategy(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> str | None:
    """Return the strategy's configured ``solutions_glob``, or ``None`` if it has none."""
    overrides = (strategy_overrides or {}).get(strategy_name, {})
    glob_pattern = overrides.get("solutions_glob")
    return glob_pattern if isinstance(glob_pattern, str) else None


def _resolve_all_samples_total(glob_pattern: str) -> int:
    """Resolve the ``_ALL_SAMPLES`` sentinel to the real file count behind a glob.

    A cheap directory listing (no ``np.loadtxt``) — used so a multi-binding
    dataset can divide "all available archive files" across its bindings the
    same way an explicit positive count is divided, instead of handing every
    binding the full archive (a cartesian-product blowup).
    """
    return len(select_archive_files(glob_pattern, count=-1, shuffle=False, seed=None, skip=0))


def _append_binding_count(
    counts_by_binding: list[dict[str, int]],
    binding_idx: int,
    strategy_name: str,
    count: int,
) -> None:
    """Accumulate one per-binding strategy count."""
    if count == _ALL_SAMPLES:
        counts_by_binding[binding_idx][strategy_name] = _ALL_SAMPLES
        return
    if count <= 0:
        return
    counts_by_binding[binding_idx][strategy_name] = (
        counts_by_binding[binding_idx].get(strategy_name, 0) + count
    )


def _allocate_strategy_counts_across_bindings(
    *,
    count: int,
    bindings: Sequence[SystemBinding],
    replacement: bool,
    rng: np.random.Generator,
) -> list[int]:
    """Distribute one strategy's global row budget across bindings."""
    num_bindings = len(bindings)
    if num_bindings < 1:
        raise ValueError("At least one binding is required for allocation.")
    if count == _ALL_SAMPLES:
        return [_ALL_SAMPLES] * num_bindings
    if count <= 0:
        return [0] * num_bindings

    if not replacement:
        base = count // num_bindings
        remainder = count % num_bindings
        return [base + (1 if idx < remainder else 0) for idx in range(num_bindings)]

    allocations = [0] * num_bindings
    sampled_indices = rng.integers(0, num_bindings, size=count)
    for binding_idx in sampled_indices.tolist():
        allocations[binding_idx] += 1
    return allocations


@dataclass(frozen=True)
class BindingAllocation:
    """Per-binding strategy budgets resolved from one global count map.

    Attributes:
        counts: Per-binding strategy count maps, parallel to the bindings.
        skips: Per-binding archive-file skip offsets, parallel to the bindings.
            Only carries entries for archive-glob-backed strategies (those with
            a ``solutions_glob`` override) — a cumulative offset into one shared
            shuffled file ordering, so each binding draws a distinct,
            non-overlapping slice of the archive instead of every binding
            reusing the same first-N files.
    """

    counts: tuple[dict[str, int], ...]
    skips: tuple[dict[str, int], ...]


def _resolve_binding_strategy_counts(
    *,
    bindings: Sequence[SystemBinding],
    spec: DatasetSpec,
    has_rhs_source: bool,
    num_matrix_samples: int,
    has_solution_source: bool = False,
) -> BindingAllocation:
    """Resolve global strategy counts into per-binding count and archive-skip maps.

    Args:
        bindings: Bindings the global budget is divided across.
        spec: Dataset assembly settings supplying the counts and replacement policy.
        has_rhs_source: Whether an RHS stream is configured.
        num_matrix_samples: Number of distinct matrix samples in the source.
        has_solution_source: Whether a per-binding ``solution_path`` stream is
            configured. When it is, archive-backed strategies receive their
            solution pre-loaded per binding (one vector, no glob read at all —
            see ``SolutionArchiveStrategy.generate``'s ``archive.lhs`` branch),
            so ``samples=-1`` is inert rather than a cartesian-product risk and
            is left as-is instead of being resolved against a glob.

    Returns:
        BindingAllocation holding the per-binding count and skip maps.
    """
    mixture = spec.mixture
    strategy_overrides = mixture.strategy_overrides
    replacement = spec.replacement
    strategy_counts = _resolve_strategy_counts(mixture.counts, mixture.mix, mixture.total)
    _validate_replacement_support(
        strategy_counts,
        replacement=replacement,
        num_matrix_samples=num_matrix_samples,
        strategy_overrides=strategy_overrides,
        has_rhs_source=has_rhs_source,
    )

    if num_matrix_samples <= 1:
        return BindingAllocation(
            counts=tuple(dict(strategy_counts) for _ in bindings),
            skips=tuple({} for _ in bindings),
        )

    rng = rng_from_seed(mixture.seed)
    counts_by_binding: list[dict[str, int]] = [{} for _ in bindings]
    skips_by_binding: list[dict[str, int]] = [{} for _ in bindings]
    for strategy_name, count in strategy_counts.items():
        glob_pattern = _archive_glob_for_strategy(strategy_name, strategy_overrides)
        resolved_count = count
        if count == _ALL_SAMPLES and not has_solution_source:
            if glob_pattern is None:
                raise ValueError(
                    f"Strategy '{strategy_name}' requested samples=-1 (\"all\") across "
                    f"{len(bindings)} matrix bindings, but has no 'solutions_glob' to "
                    "resolve a real total from. Replicating -1 to every binding would "
                    "load the full archive once per binding (a cartesian-product "
                    "blowup) — set an explicit positive 'samples' count instead."
                )
            resolved_count = _resolve_all_samples_total(glob_pattern)

        allocations = _allocate_strategy_counts_across_bindings(
            count=resolved_count,
            bindings=bindings,
            replacement=replacement,
            rng=rng,
        )
        cumulative_skip = 0
        for binding_idx, allocated_count in enumerate(allocations):
            _append_binding_count(counts_by_binding, binding_idx, strategy_name, allocated_count)
            if glob_pattern is not None and allocated_count > 0:
                skips_by_binding[binding_idx][strategy_name] = cumulative_skip
            cumulative_skip += allocated_count
    return BindingAllocation(counts=tuple(counts_by_binding), skips=tuple(skips_by_binding))


@dataclass(frozen=True)
class _BindingResult:
    """Result from processing one binding.

    Stores generated and accumulated data from a single matrix-RHS binding.

    Attributes:
        rhs_block: Generated RHS block for this binding
        solution_block: Generated solution block for this binding
        matrix_norm_value: Computed norm of normalized matrix
        matrix_value_scale: Scaling factor applied to matrix values
        scale_params: Dictionary of scale parameters or None
    """

    rhs_block: np.ndarray
    solution_block: np.ndarray
    row_kind_codes: np.ndarray
    matrix_sample_index: np.ndarray
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


def _merge_binding_skip_overrides(
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    strategy_skips: Mapping[str, int] | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    """Add each strategy's per-binding archive skip on top of its configured base skip.

    Never mutates ``strategy_overrides`` — it's shared across every binding's call.
    """
    if not strategy_skips:
        return strategy_overrides
    merged = {name: dict(opts) for name, opts in (strategy_overrides or {}).items()}
    for strategy_name, binding_skip in strategy_skips.items():
        opts = merged.setdefault(strategy_name, {})
        opts["skip"] = opts.get("skip", 0) + binding_skip
    return merged


def _process_binding(
    binding: SystemBinding,
    streams: OpenedStreams,
    get_matrix: Callable[[int], _CachedMatrix],
    mixture: MixtureSpec,
) -> _BindingResult:
    """Process one matrix-RHS binding and generate samples.

    Args:
        binding: Single binding to process.
        streams: Opened source streams the binding draws its RHS/solution from.
        get_matrix: Callable ``(sample_id: int) -> _CachedMatrix``; caches internally.
        mixture: Strategy budgets and RNG controls already narrowed to this binding.

    Returns:
        BindingResult with generated data blocks
    """
    rhs_stream = streams.rhs
    solution_stream = streams.solution
    cached = get_matrix(binding.matrix_sample_id)

    # Load optional RHS
    single_rhs: np.ndarray | None = None
    if rhs_stream is not None and binding.rhs_sample_id is not None:
        rhs_sample = rhs_stream.load_sample(binding.rhs_sample_id)
        single_rhs = np.asarray(rhs_sample.vector, dtype=np.float64)
        if single_rhs.shape[0] != cached.matrix_norm.shape[0]:
            raise ValueError(
                f"RHS sample {binding.rhs_sample_id} length {single_rhs.shape[0]} "
                f"doesn't match matrix size {cached.matrix_norm.shape[0]}"
            )
        if cached.scale is not None:
            single_rhs = cached.scale.scale_rhs(single_rhs)

    # Load optional per-binding solution
    single_solution: np.ndarray | None = None
    if solution_stream is not None and binding.solution_sample_id is not None:
        sol_sample = solution_stream.load_sample(binding.solution_sample_id)
        single_solution = np.asarray(sol_sample.vector, dtype=np.float64)

    # Generate mixture
    logger.info(
        f"Generating/loading samples for binding sample_id={binding.sample_id} "
        f"(matrix_id={binding.matrix_sample_id})..."
    )
    generated = _generate_mixture_with_metadata(
        cached.matrix_norm,
        mixture,
        single_rhs=single_rhs,
        single_solution=single_solution,
    )
    X_final = generated.rhs
    Y_final = generated.solutions
    row_kind_codes = generated.row_kind_codes

    if X_final.shape != Y_final.shape:
        raise ValueError(
            f"Generated RHS/solution shape mismatch: {X_final.shape} vs {Y_final.shape}"
        )

    return _BindingResult(
        rhs_block=np.asarray(X_final, dtype=np.float64),
        solution_block=np.asarray(Y_final, dtype=np.float64),
        row_kind_codes=np.asarray(row_kind_codes, dtype=np.uint8),
        matrix_sample_index=np.full(X_final.shape[0], binding.matrix_sample_id, dtype=np.int64),
        matrix_norm_value=float(cached.matrix_norm_value),
        matrix_value_scale=float(cached.matrix_value_scale),
        scale_params=cached.scale_params,
    )


def _resolve_final_scale(
    norm_values: list[float],
    scale_values: list[float],
    metadata_values: list[ScaleMetadata | None],
) -> tuple[float, float, ScaleMetadata | None]:
    """Resolve manifest-level normalization metadata from all bindings.

    Matrix samples are normalized per binding before they are written. The
    manifest, however, exposes only one dataset-level normalization block.
    When bindings disagree on scale details, the manifest omits ambiguous scale
    metadata instead of inventing a shared reversible scale.

    Args:
        norm_values: Matrix norm values from each binding.
        scale_values: Matrix value scale factors from each binding.
        metadata_values: Scale metadata from each binding.

    Returns:
        Tuple of ``(final_matrix_norm, final_matrix_scale, final_scale_metadata)``.
        The returned ``final_scale_metadata`` is ``None`` when a multi-binding
        dataset does not share one exact scale payload.
    """
    if not norm_values or not scale_values:
        raise ValueError("No norm or scale values to resolve")

    # Resolve matrix norm
    matrix_norm_value = float(norm_values[0])
    if not all(
        np.isclose(v, matrix_norm_value, rtol=_NORM_AGREEMENT_RTOL, atol=_NORM_AGREEMENT_ATOL)
        for v in norm_values
    ):
        logger.warning(
            "Bindings produced different normalized matrix norms; "
            "the dataset manifest keeps a representative matrix_norm only."
        )

    # Resolve matrix value scale
    matrix_value_scale = float(scale_values[0])
    if not all(
        np.isclose(v, matrix_value_scale, rtol=_NORM_AGREEMENT_RTOL, atol=_NORM_AGREEMENT_ATOL)
        for v in scale_values
    ):
        logger.warning(
            "Bindings produced different matrix value scales; "
            "the dataset manifest omits a shared reversible matrix scale."
        )
        matrix_value_scale = 1.0

    # Resolve scale metadata
    unique_scale_payloads = {
        json.dumps(payload, sort_keys=True) if payload is not None else "null"
        for payload in metadata_values
    }
    scale_metadata: ScaleMetadata | None = None
    if len(unique_scale_payloads) == 1:
        scale_metadata = metadata_values[0]
    else:
        logger.warning(
            "Bindings produced different scale metadata payloads; "
            "the dataset manifest intentionally stores no shared scale metadata."
        )
        scale_metadata = None

    return matrix_norm_value, matrix_value_scale, scale_metadata


@dataclass(frozen=True)
class _GenerationRunContext:
    """Opened streams plus the per-binding budgets resolved against them.

    Attributes:
        streams: Every opened source stream and the resolved bindings.
        allocation: Per-binding strategy count and archive-skip maps.
    """

    streams: OpenedStreams
    allocation: BindingAllocation


def _prepare_generation_context(
    source: SourceSpec,
    spec: DatasetSpec,
) -> _GenerationRunContext:
    """Open all source streams, bind them, and resolve per-binding strategy counts.

    Args:
        source: Resolved source paths and per-stream sample filters.
        spec: Dataset assembly settings supplying the strategy budgets.

    Returns:
        _GenerationRunContext pairing the opened streams with their allocation.
    """
    streams = _open_streams(source)
    allocation = _resolve_binding_strategy_counts(
        bindings=streams.bindings,
        spec=spec,
        has_rhs_source=streams.rhs is not None,
        num_matrix_samples=len(streams.matrix.sample_ids),
        has_solution_source=streams.solution is not None,
    )
    return _GenerationRunContext(streams=streams, allocation=allocation)


@dataclass(frozen=True)
class AccumulatedBindings:
    """Everything collected while processing every binding of one generation run.

    Each list is parallel to the bindings that actually emitted samples, in
    processing order; ``param_blocks`` is indexed by parameter stream first.

    Attributes:
        rhs_blocks: Generated RHS block per emitting binding.
        solution_blocks: Generated solution block per emitting binding.
        row_kind_blocks: Row-kind code block per emitting binding.
        matrix_sample_index_blocks: Matrix-sample-id block per emitting binding.
        param_blocks: Per parameter stream, the tiled block per emitting binding.
        matrix_norm_values: Normalized matrix norm per emitting binding.
        matrix_value_scale_values: Matrix value scale factor per emitting binding.
        scale_metadata_values: Scale metadata payload per emitting binding.
        emitted_binding_count: Number of bindings that emitted at least one sample.
    """

    rhs_blocks: list[np.ndarray]
    solution_blocks: list[np.ndarray]
    row_kind_blocks: list[np.ndarray]
    matrix_sample_index_blocks: list[np.ndarray]
    param_blocks: list[list[np.ndarray]]
    matrix_norm_values: list[float]
    matrix_value_scale_values: list[float]
    scale_metadata_values: list[ScaleMetadata | None]
    emitted_binding_count: int


def _accumulate_bindings(
    *,
    context: _GenerationRunContext,
    get_matrix: Callable[[int], _CachedMatrix],
    accumulator: DenseAccumulatorPort,
    mixture: MixtureSpec,
) -> AccumulatedBindings:
    """Process all bindings and accumulate rhs/solution blocks, param blocks, and scale metadata.

    Args:
        context: Opened streams paired with their per-binding allocation.
        get_matrix: Callable that loads and caches a normalized matrix by sample ID.
        accumulator: Dataset accumulator for writing matrix samples.
        mixture: Global mixture settings; each binding runs with its own counts
            and archive-skip offsets layered on top.

    Returns:
        AccumulatedBindings holding every per-binding block and scale value.
    """
    streams = context.streams
    param_streams = streams.parameters
    single_matrix_mode = streams.single_matrix_mode
    single_matrix_written = False
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    row_kind_blocks: list[np.ndarray] = []
    matrix_sample_index_blocks: list[np.ndarray] = []
    param_blocks: list[list[np.ndarray]] = [[] for _ in param_streams]
    matrix_norm_values: list[float] = []
    matrix_value_scale_values: list[float] = []
    scale_metadata_values: list[ScaleMetadata | None] = []
    emitted_binding_count = 0

    for binding, binding_strategy_counts, binding_strategy_skips in zip(
        streams.bindings, context.allocation.counts, context.allocation.skips, strict=True
    ):
        if not binding_strategy_counts:
            continue
        binding_mixture = replace(
            mixture,
            counts=binding_strategy_counts,
            mix=None,
            total=None,
            strategy_overrides=_merge_binding_skip_overrides(
                mixture.strategy_overrides, binding_strategy_skips
            ),
        )
        result = _process_binding(binding, streams, get_matrix, binding_mixture)

        n_samples = result.rhs_block.shape[0]
        if n_samples == 0:
            continue

        rhs_blocks.append(result.rhs_block)
        solution_blocks.append(result.solution_block)
        row_kind_blocks.append(result.row_kind_codes)
        matrix_sample_index_blocks.append(result.matrix_sample_index)
        emitted_binding_count += 1

        for k, (stream, sample_id) in enumerate(zip(param_streams, binding.parameters_sample_ids)):
            if sample_id is not None:
                vec = stream.load_sample(sample_id).vector
                param_blocks[k].append(np.tile(vec, (n_samples, 1)))

        cached = get_matrix(binding.matrix_sample_id)
        if single_matrix_mode:
            if not single_matrix_written:
                accumulator.append_dense_matrix(cached.matrix_norm, repeats=1)
                single_matrix_written = True
        else:
            accumulator.append_dense_matrix(cached.matrix_norm, repeats=int(n_samples))

        matrix_norm_values.append(result.matrix_norm_value)
        matrix_value_scale_values.append(result.matrix_value_scale)
        scale_metadata_values.append(result.scale_params)

    return AccumulatedBindings(
        rhs_blocks=rhs_blocks,
        solution_blocks=solution_blocks,
        row_kind_blocks=row_kind_blocks,
        matrix_sample_index_blocks=matrix_sample_index_blocks,
        param_blocks=param_blocks,
        matrix_norm_values=matrix_norm_values,
        matrix_value_scale_values=matrix_value_scale_values,
        scale_metadata_values=scale_metadata_values,
        emitted_binding_count=emitted_binding_count,
    )


def _finalize_payload(
    *,
    accumulated: AccumulatedBindings,
    accumulator: DenseAccumulatorPort,
    spec: DatasetSpec,
    layout: LayoutType = LayoutType.MANY_MATRICES,
) -> GeneratedDatasetPayload:
    """Stack arrays, resolve scale, finalize accumulator, and build the payload.

    Args:
        accumulated: Per-binding blocks and scale values gathered while processing.
        accumulator: Dataset accumulator to finalize.
        spec: Dataset assembly settings supplying the normalization metadata.
        layout: Matrix storage layout recorded in the payload.

    Returns:
        Immutable GeneratedDatasetPayload ready for persistence.

    Raises:
        ValueError: If no samples were generated or accumulator is empty.
    """
    if not accumulated.rhs_blocks or not accumulated.solution_blocks:
        raise ValueError("No samples were generated for dataset persistence.")

    parameters_arrays: tuple[np.ndarray, ...] = tuple(
        np.vstack(blocks) for blocks in accumulated.param_blocks if blocks
    )
    rhs_all = np.vstack(accumulated.rhs_blocks)
    solutions_all = np.vstack(accumulated.solution_blocks)
    row_kind_codes = (
        np.concatenate(accumulated.row_kind_blocks)
        if accumulated.row_kind_blocks
        else np.empty((0,), dtype=np.uint8)
    )
    matrix_sample_index = (
        np.concatenate(accumulated.matrix_sample_index_blocks)
        if accumulated.matrix_sample_index_blocks
        else np.empty((0,), dtype=np.int64)
    )
    if row_kind_codes.shape[0] != rhs_all.shape[0]:
        raise ValueError(
            f"row_kind metadata has {row_kind_codes.shape[0]} entries "
            f"but RHS matrix has {rhs_all.shape[0]} rows — "
            "counts must match. A generation strategy may have produced a "
            "mismatched number of samples."
        )
    if matrix_sample_index.shape[0] != rhs_all.shape[0]:
        raise ValueError(
            f"matrix_sample_index has {matrix_sample_index.shape[0]} entries "
            f"but RHS matrix has {rhs_all.shape[0]} rows — "
            "counts must match. A generation strategy may have produced a "
            "mismatched number of samples."
        )
    matrix_artifact_path = accumulator.finalize()
    matrix_size = accumulator.matrix_size

    if matrix_size is None:
        raise ValueError("Accumulator is empty — no matrix samples were written.")

    matrix_norm_value, matrix_value_scale, scale_metadata = _resolve_final_scale(
        accumulated.matrix_norm_values,
        accumulated.matrix_value_scale_values,
        accumulated.scale_metadata_values,
    )
    return GeneratedDatasetPayload(
        rhs=rhs_all,
        solutions=solutions_all,
        matrix_artifact_path=matrix_artifact_path,
        matrix_size=matrix_size,
        normalization_type=str(spec.normalize),
        matrix_norm=matrix_norm_value,
        matrix_norm_type=spec.matrix_norm_type,
        matrix_value_scale=matrix_value_scale,
        scale_metadata=scale_metadata,
        num_bindings=accumulated.emitted_binding_count,
        parameters_arrays=parameters_arrays,
        layout=layout,
        row_kind_codes=row_kind_codes,
        matrix_sample_index=matrix_sample_index,
    )


def build_dataset_payload(
    source: SourceSpec,
    spec: DatasetSpec,
    *,
    accumulator: DenseAccumulatorPort,
) -> GeneratedDatasetPayload:
    """Build an in-memory dataset payload from streamed matrix sources.

    Orchestrates the generation pipeline by coordinating stream opening, matrix
    normalization, per-binding sample generation, and payload assembly.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: How the dataset is assembled — strategy budgets, RNG controls,
            replacement policy and normalization.
        accumulator: Dataset accumulator for writing matrix samples.

    Returns:
        Immutable generated dataset payload ready for persistence.
    """
    logger.info("Building dataset...")
    logger.info(f"  Matrix: {source.matrix_path}")
    if source.rhs_path is not None:
        logger.info(f"  RHS source: {source.rhs_path}")
    if source.solution_path is not None:
        logger.info(f"  Solution stream: {source.solution_path}")
    for i, pp in enumerate(source.parameters_paths):
        logger.info(f"  Parameters stream [{i}]: {pp}")

    context = _prepare_generation_context(source, spec)
    matrix_stream = context.streams.matrix
    logger.info(
        f"  Matrix samples: {len(matrix_stream.sample_ids)} | "
        f"System bindings: {len(context.streams.bindings)}"
    )
    logger.info(f"  Normalization: {spec.normalize}")

    @cache
    def _get_matrix(sample_id: int) -> _CachedMatrix:
        from neuralls.domain.linalg import calculate_matrix_norm

        from .helpers import _normalize_matrix_for_generation

        dense_sample = matrix_stream.load_dense_sample(sample_id)
        dense_matrix = np.asarray(dense_sample.matrix, dtype=np.float64)
        if dense_matrix.shape[0] != dense_matrix.shape[1]:
            raise ValueError(f"Matrix sample {sample_id} must be square, got {dense_matrix.shape}")
        matrix_norm, scale, matrix_value_scale = _normalize_matrix_for_generation(
            dense_matrix,
            spec.normalize,
            spectral_radius_bound=None,
        )
        matrix_norm_value = calculate_matrix_norm(matrix_norm, norm_type=spec.matrix_norm_type)
        scale_params = serialize_scale_metadata(scale)
        return _CachedMatrix(
            matrix_norm=matrix_norm,
            scale=scale,
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )

    layout = (
        LayoutType.BROADCAST_SINGLE
        if context.streams.single_matrix_mode
        else LayoutType.MANY_MATRICES
    )
    accumulated = _accumulate_bindings(
        context=context,
        get_matrix=_get_matrix,
        accumulator=accumulator,
        mixture=spec.mixture,
    )
    payload = _finalize_payload(
        accumulated=accumulated,
        accumulator=accumulator,
        spec=spec,
        layout=layout,
    )
    logger.info(
        "Dataset payload built successfully: "
        f"samples={payload.rhs.shape[0]}, "
        f"matrix_samples={accumulated.emitted_binding_count}, "
        f"parameter_streams={len(payload.parameters_arrays)}"
    )
    return payload


__all__ = [
    "_shuffle_samples",
    "build_dataset_payload",
    "generate_mixture",
]
