"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pydantic import ValidationError

from neuralls.domain.normalization import IScale
from neuralls.domain.normalization import matrix_norm as system_matrix_norm
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import (
    GenerationStrategyKind,
    MatrixFormat,
    MatrixNormType,
    RowKind,
    ScaleMetadata,
    SystemMatrix,
)

from .allocation import archive_units, split_remainder
from .batch import SampleBatch
from .batch_generator import StrategyRunner, generate_batches
from .batch_plan import ALL_SAMPLES, BatchPlan, BindingAllocation, plan_batches
from .helpers import (
    derive_seed,
    derive_strategy_seed,
    normalize_matrix_for_generation,
    resolve_strategy_counts,
    select_archive_files,
    serialize_scale_metadata,
    solution_row_count,
)
from .interfaces import ArchiveData
from .runner import GeneratedSamples, rows_per_base_system, run_generation
from .scalar_aggregate import BindingScale, ScalarAggregator
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
    """Compile-time properties for a known generation strategy.

    Attributes:
        is_archive: Whether the strategy draws from a finite pool of files, each
            usable at most once. Archives never overload a matrix.
        glob_key: Override key holding the strategy's archive glob, or ``None`` when the
            strategy has no file-backed archive. The glob fixes the pool that explicit
            (matrix, file) indices refer to.
    """

    is_archive: bool
    glob_key: str | None = None


_STRATEGY_PROPERTIES: dict[GenerationStrategyKind, _StrategyProperties] = {
    GenerationStrategyKind.SOLUTION_ARCHIVE: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.RHS_ARCHIVE: _StrategyProperties(is_archive=True, glob_key="rhs_glob"),
    GenerationStrategyKind.SCALED_SOLUTIONS: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.VALIDATED_ARCHIVE: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.RESIDUALS: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.GAUSSIAN_RESIDUALS: _StrategyProperties(is_archive=False),
}


def _properties_for(strategy_name: str) -> _StrategyProperties | None:
    """Properties of a known strategy, or ``None`` for a name outside the enum."""
    try:
        return _STRATEGY_PROPERTIES.get(GenerationStrategyKind(strategy_name))
    except ValueError:
        return None


@dataclass(frozen=True)
class _StrategyRows:
    """One strategy's training pairs on one system, in generation order.

    Attributes:
        rhs: Feature rows (the RHS, or the trace residuals), shape (rows, n).
        solutions: Target rows (the solutions, or the trace errors), shape (rows, n).
        row_kind_codes: Per-row ``RowKind`` codes, shape (rows,).
    """

    rhs: np.ndarray
    solutions: np.ndarray
    row_kind_codes: np.ndarray


def _row_kind_codes_for(strategy_name: str, generated: GeneratedSamples) -> np.ndarray:
    """Classify each row a strategy emitted, using its trace iteration indices when present."""
    strategy_kind = GenerationStrategyKind(strategy_name)
    semantic_size = 0
    if generated.error_traces is not None:
        semantic_size = int(generated.error_traces.errors.shape[0])
    elif generated.residual_traces is not None:
        semantic_size = int(generated.residual_traces.residuals.shape[0])
    elif generated.rhs is not None:
        semantic_size = int(generated.rhs.shape[0])
    if semantic_size == 0:
        return encode_row_kind_array([])

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
    return encode_row_kind_array(row_kinds)


def _generate_strategy_rows(
    A: SystemMatrix,
    mixture: MixtureSpec,
    strategy_name: str,
    count: int,
    *,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
    single_solution: np.ndarray | None = None,
) -> _StrategyRows | None:
    """Run one strategy for ``count`` samples and return its pairs and row kinds.

    Args:
        A: System matrix the strategy solves against, shape (n, n).
        mixture: Strategy overrides, solver overrides and the seed the strategy seed derives from.
        strategy_name: Registered strategy to run.
        count: Sample budget passed to the strategy as its ``samples`` option.
        archive_solutions: Pre-computed solutions for archive-based generation.
        archive_rhs: Pre-computed RHS vectors for archive-based generation.
        single_rhs: Single RHS vector for single-RHS (trace) strategies.
        single_solution: Single pre-computed solution used as a one-row archive.

    Returns:
        The strategy's rows, or ``None`` when it produced neither pairs nor traces.

    Raises:
        ValueError: If the strategy configuration is invalid or its rhs/solutions disagree.
    """
    cfg = dict((mixture.strategy_overrides or {}).get(strategy_name, {}))
    cfg.setdefault("seed", derive_strategy_seed(mixture.seed, strategy_name))
    cfg["samples"] = count

    effective_archive_solutions = archive_solutions
    if single_solution is not None and archive_solutions is None:
        effective_archive_solutions = single_solution.reshape(1, -1)

    archive_data: ArchiveData | None = None
    if effective_archive_solutions is not None:
        archive_data = ArchiveData(lhs=effective_archive_solutions, rhs=archive_rhs)

    # Pydantic (extra="forbid") will raise ValidationError on unknown keys — fail fast.
    try:
        generated = run_generation(
            strategy_name,
            A,
            cfg=cfg,
            solver=(
                mixture.solver_overrides.get(strategy_name) if mixture.solver_overrides else None
            ),
            archive=archive_data,
            single_rhs=single_rhs,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for strategy '{strategy_name}': {e}") from e

    row_kind_codes = _row_kind_codes_for(strategy_name, generated)
    # Trace strategies expose their training pairs via trace structs, not rhs/solutions.
    if generated.error_traces is not None:
        return _StrategyRows(
            rhs=generated.error_traces.residuals,  # r_k
            solutions=generated.error_traces.errors,  # e_k = x_true - x_k
            row_kind_codes=row_kind_codes,
        )
    if generated.residual_traces is not None:
        return _StrategyRows(
            rhs=generated.residual_traces.residuals,  # A @ p_k
            solutions=generated.residual_traces.solutions,  # p_k
            row_kind_codes=row_kind_codes,
        )
    if (generated.rhs is None) != (generated.solutions is None):
        raise ValueError(
            f"Strategy '{strategy_name}' returned rhs and solutions with inconsistent "
            f"None-ness: rhs={'None' if generated.rhs is None else 'array'}, "
            f"solutions={'None' if generated.solutions is None else 'array'}."
        )
    if generated.rhs is None or generated.solutions is None:
        return None
    return _StrategyRows(
        rhs=generated.rhs,
        solutions=generated.solutions,
        row_kind_codes=row_kind_codes,
    )


@dataclass(frozen=True)
class _CachedMatrix:
    """Immutable cached matrix data for generation.

    Stores all derived matrices and scaling information computed once
    per unique matrix_sample_id, avoiding redundant computation.

    Attributes:
        matrix_norm: Normalized system matrix, in the dataset's matrix format
            (dense ndarray or CSR; CSR is never densified)
        scale: IScale object or None (scaling strategy applied)
        matrix_norm_value: Computed matrix norm value
        matrix_value_scale: Scaling factor applied
        scale_params: Dictionary of scale parameters or None
    """

    matrix_norm: SystemMatrix
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


def _open_streams(source: SourceSpec, *, solution_unbound: bool = False) -> OpenedStreams:
    """Open matrix, optional RHS, optional solution, and optional parameter streams and bind them.

    Args:
        source: Resolved source paths and per-stream sample filters.
        solution_unbound: Whether an explicit solution file supplies its rows to the
            bindings by row position (``samples=-1`` or an explicit archive count). Its
            ids then take no part in binding, so they are not matched against matrix ids.

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
        solution_ids=(
            solution_stream.sample_ids
            if solution_stream is not None and not solution_unbound
            else None
        ),
        parameters_ids_list=tuple(s.sample_ids for s in param_streams),
    )
    return OpenedStreams(
        matrix=matrix_stream,
        rhs=rhs_stream,
        solution=solution_stream,
        parameters=param_streams,
        bindings=tuple(bindings),
    )


def _replacement_rejection_message(strategy_name: str) -> str:
    """Explain why ``replacement = true`` is rejected for one strategy."""
    props = _properties_for(strategy_name)
    if props is not None and props.is_archive:
        return (
            f"Strategy '{strategy_name}' draws from a finite archive where each file is "
            "used at most once, so replacement = true is not supported."
        )
    return (
        "replacement = true is not supported: generated strategies overload each matrix "
        "uniformly, which replaces random replacement."
    )


def _reject_replacement_request(strategy_counts: Mapping[str, int], *, replacement: bool) -> None:
    """Fail fast when replacement is requested for any active strategy."""
    # count != 0 (not "> 0") so ALL_SAMPLES (-1, "emit everything") still counts as active.
    active = [name for name, count in strategy_counts.items() if count != 0]
    if replacement and active:
        raise ValueError(_replacement_rejection_message(active[0]))


def _archive_glob_for_strategy(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> str | None:
    """Return the strategy's configured archive glob, or ``None`` if it has none."""
    props = _properties_for(strategy_name)
    if props is None or props.glob_key is None:
        return None
    overrides = (strategy_overrides or {}).get(strategy_name, {})
    glob_pattern = overrides.get(props.glob_key)
    return glob_pattern if isinstance(glob_pattern, str) else None


def _glob_key_phrase(strategy_name: str) -> str:
    """Name the override key holding this strategy's glob, quoted for error messages."""
    props = _properties_for(strategy_name)
    if props is None or props.glob_key is None:
        return "archive glob"
    return f"'{props.glob_key}'"


def _archive_pool_size(glob_pattern: str, skip: int) -> int:
    """Count the archive files a strategy draws from after its configured skip.

    A cheap directory listing (no ``np.loadtxt``). Explicit archive indices refer
    to positions in this pool, so its size is the ``K`` of the (matrix, file) grid.
    """
    return len(select_archive_files(glob_pattern, count=-1, shuffle=False, seed=None, skip=skip))


def _binding_indices_by_matrix(bindings: Sequence[SystemBinding]) -> dict[int, list[int]]:
    """Group binding positions by the matrix sample each binding applies."""
    indices_by_matrix: dict[int, list[int]] = {}
    for binding_idx, binding in enumerate(bindings):
        indices_by_matrix.setdefault(binding.matrix_sample_id, []).append(binding_idx)
    return indices_by_matrix


def _archive_skip(
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None, strategy_name: str
) -> int:
    """Return the strategy's configured archive skip, defaulting to zero."""
    opts = (strategy_overrides or {}).get(strategy_name, {})
    return int(opts.get("skip", 0))


def _split_generated_count(
    count: int,
    groups: Sequence[Sequence[int]],
    num_bindings: int,
    seed: int,
) -> list[int]:
    """Split one generated count over matrices, then over each matrix's bindings.

    Every level uses the remainder allocation, so nothing is dropped. Pure given the seed.
    """
    allocations = [0] * num_bindings
    per_matrix = split_remainder(count, len(groups), seed=derive_seed(seed, "matrix-split"))
    for matrix_idx, (group, matrix_count) in enumerate(zip(groups, per_matrix, strict=True)):
        per_binding = split_remainder(
            int(matrix_count), len(group), seed=derive_seed(seed, "binding-split", matrix_idx)
        )
        for binding_idx, binding_count in zip(group, per_binding, strict=True):
            allocations[binding_idx] = int(binding_count)
    return allocations


def _rows_per_system_for(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    samples: int,
) -> int:
    """K_rows for one strategy's budget: rows one of its base systems yields."""
    opts = dict((strategy_overrides or {}).get(strategy_name, {}))
    return rows_per_base_system(strategy_name, {**opts, "samples": samples})


def _split_generated_rows(
    rows: int,
    rows_per_system: int,
    groups: Sequence[Sequence[int]],
    num_bindings: int,
    seed: int,
) -> list[int]:
    """Split a generated row budget over base systems, then convert back to rows.

    The budget becomes B = ceil(rows / K_rows) base systems, split like any count.
    The overshoot O = B * K_rows - rows is trimmed from the lowest-index owner of a
    base system, which always keeps at least one row since O < K_rows.
    """
    base_systems = math.ceil(rows / rows_per_system)
    units = _split_generated_count(base_systems, groups, num_bindings, seed)
    row_counts = [units_b * rows_per_system for units_b in units]
    overshoot = base_systems * rows_per_system - rows
    if overshoot:
        first_owner = next(idx for idx, units_b in enumerate(units) if units_b > 0)
        row_counts[first_owner] -= overshoot
    return row_counts


def _allocate_archive_rows(
    *,
    strategy_name: str,
    glob_pattern: str,
    skip: int,
    rows: int,
    rows_per_system: int,
    groups: Sequence[Sequence[int]],
    num_bindings: int,
) -> tuple[list[int], list[tuple[int, ...]]]:
    """Assign archive (matrix, file) base-system units to bindings so no pair repeats.

    The row request becomes base systems, capped at M*K. The cap is reported once,
    in rows and base systems. Overshoot rows are trimmed from the last-unit owner.
    Requires one binding per matrix (checked by the caller).

    Returns:
        Per-binding row counts and per-binding explicit file indices, both parallel to the bindings.
    """
    num_matrices = len(groups)
    num_files = _archive_pool_size(glob_pattern, skip)
    capacity = num_matrices * num_files
    requested_rows = capacity * rows_per_system if rows == ALL_SAMPLES else rows
    requested_base = math.ceil(requested_rows / rows_per_system)
    emitted_base = min(requested_base, capacity)
    if requested_base > capacity:
        logger.warning(
            f"Strategy '{strategy_name}' requested {requested_rows} rows "
            f"({requested_base} base systems) but the archive has {capacity} (matrix, file) "
            f"pairs: emitting {emitted_base * rows_per_system} rows ({emitted_base} base systems)."
        )
    units = archive_units(num_matrices, num_files, emitted_base)

    files_by_matrix: list[list[int]] = [[] for _ in groups]
    for matrix_pos, file_idx in units:
        files_by_matrix[matrix_pos].append(file_idx)

    row_by_matrix = [len(matrix_files) * rows_per_system for matrix_files in files_by_matrix]
    overshoot = emitted_base * rows_per_system - requested_rows
    if units and overshoot > 0:
        last_unit_matrix, _ = units[-1]
        row_by_matrix[last_unit_matrix] -= overshoot

    counts = [0] * num_bindings
    files: list[tuple[int, ...]] = [() for _ in range(num_bindings)]
    for group, matrix_rows, matrix_files in zip(
        groups, row_by_matrix, files_by_matrix, strict=True
    ):
        binding_idx = group[0]
        counts[binding_idx] = matrix_rows
        files[binding_idx] = tuple(matrix_files)
    return counts, files


def _file_backed_archive_globs(
    strategy_counts: Mapping[str, int],
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    *,
    has_solution_source: bool,
) -> dict[str, str]:
    """Archive strategies with a glob that get explicit per-binding file indices.

    A per-binding solution source and an archive glob are two competing pools for the
    same archive strategy, and the glob cannot be assigned per binding alongside it,
    so combining them is rejected rather than letting files repeat.

    Raises:
        ValueError: If ``has_solution_source`` is set and an active archive strategy
            also has a glob.
    """
    globs: dict[str, str] = {}
    for strategy_name, count in strategy_counts.items():
        if count == 0:
            continue
        glob_pattern = _archive_glob_for_strategy(strategy_name, strategy_overrides)
        if glob_pattern is None:
            continue
        if has_solution_source:
            raise ValueError(
                f"Strategy '{strategy_name}' has a glob ({glob_pattern!r}) and a per-binding "
                "solution_path source at the same time: the glob pool cannot be assigned "
                "without repeats alongside it. Remove the glob or the solution_path."
            )
        globs[strategy_name] = glob_pattern
    return globs


def _reject_repeated_archive_bindings(
    archive_names: Sequence[str],
    bindings: Sequence[SystemBinding],
) -> None:
    """Reject archive strategies whose matrix has several bindings.

    Each such binding would draw the same archive pool, so the same (matrix, file)
    pair would be emitted once per binding.
    """
    if not archive_names:
        return
    for matrix_id, binding_indices in _binding_indices_by_matrix(bindings).items():
        if len(binding_indices) > 1:
            raise ValueError(
                f"Archive strategies {list(archive_names)} draw from one shared pool per matrix, "
                f"but matrix {matrix_id} has {len(binding_indices)} bindings: "
                "files would repeat across bindings. Use one binding per matrix."
            )


def _single_matrix_allocation(
    bindings: Sequence[SystemBinding],
    strategy_counts: Mapping[str, int],
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    archive_globs: Mapping[str, str],
    solution_rows_total: int | None,
    cyclic_solution_strategies: frozenset[str],
) -> BindingAllocation:
    """Allocation when the source has one matrix: every binding takes the global counts.

    An open-ended (``samples = -1``) archive strategy is sized from its glob, since its
    pool is a single matrix's K files. With an explicit solution file, an open-ended
    strategy takes the file's row count on every binding, and an explicit count of a
    solution-archive strategy is capped at that row count. Other counts pass through.
    """
    counts: dict[str, int] = dict(strategy_counts)
    if solution_rows_total is not None:
        counts = {
            name: _solution_file_count(name, count, solution_rows_total)
            if count == ALL_SAMPLES or name in cyclic_solution_strategies
            else count
            for name, count in counts.items()
        }
    files: dict[str, tuple[int, ...]] = {}
    for strategy_name, glob_pattern in archive_globs.items():
        if strategy_counts[strategy_name] != ALL_SAMPLES:
            continue
        sized_counts, sized_files = _allocate_archive_rows(
            strategy_name=strategy_name,
            glob_pattern=glob_pattern,
            skip=_archive_skip(strategy_overrides, strategy_name),
            rows=ALL_SAMPLES,
            rows_per_system=_rows_per_system_for(strategy_name, strategy_overrides, ALL_SAMPLES),
            groups=[[0]],
            num_bindings=1,
        )
        counts[strategy_name] = sized_counts[0]
        files[strategy_name] = sized_files[0]
    return BindingAllocation(
        counts=tuple(dict(counts) for _ in bindings),
        file_indices=tuple(dict(files) for _ in bindings),
    )


def _resolve_binding_strategy_counts(
    *,
    bindings: Sequence[SystemBinding],
    spec: DatasetSpec,
    num_matrix_samples: int,
    has_solution_source: bool = False,
    solution_rows_total: int | None = None,
) -> BindingAllocation:
    """Resolve global strategy counts into per-binding counts and archive file indices.

    Generated counts are split across matrices, then across each matrix's bindings,
    with the remainder allocation, so no sample is dropped. Archive counts are mapped
    onto the (matrix, file) grid with ``archive_units``, so no pair is emitted twice.

    Args:
        bindings: Bindings the global budget is divided across.
        spec: Dataset assembly settings supplying the counts and replacement policy.
        num_matrix_samples: Number of distinct matrix samples in the source.
        has_solution_source: Whether a ``solution_path`` stream is configured. Archive
            strategies then take no glob of their own.
        solution_rows_total: Rows the ``solution_path`` source supplies by position, or
            ``None`` when it binds by sample id or is absent. For an explicit file this is
            its row count; for a glob it is the number of matched files. Listing only:
            no content is read. ``samples=-1`` takes all of them on every binding, and an
            explicit solution-archive count is drawn cyclically from them.

    Returns:
        BindingAllocation holding the per-binding count and file-index maps.

    Raises:
        ValueError: If an archive strategy has several bindings on one matrix, if
            ``samples=-1`` has no glob or solution file to resolve against, or if
            replacement is requested.
    """
    mixture = spec.mixture
    strategy_overrides = mixture.strategy_overrides
    strategy_counts = resolve_strategy_counts(mixture.counts, mixture.mix, mixture.total)
    _reject_replacement_request(strategy_counts, replacement=spec.replacement)

    archive_globs = _file_backed_archive_globs(
        strategy_counts, strategy_overrides, has_solution_source=has_solution_source
    )
    _reject_repeated_archive_bindings(list(archive_globs), bindings)
    cyclic_solution_strategies = _explicit_solution_strategies(strategy_counts)

    if num_matrix_samples <= 1:
        return _single_matrix_allocation(
            bindings,
            strategy_counts,
            strategy_overrides,
            archive_globs,
            solution_rows_total,
            cyclic_solution_strategies,
        )

    groups = list(_binding_indices_by_matrix(bindings).values())
    counts_by_binding: list[dict[str, int]] = [{} for _ in bindings]
    files_by_binding: list[dict[str, tuple[int, ...]]] = [{} for _ in bindings]
    for strategy_name, count in strategy_counts.items():
        if count == 0:
            continue
        rows_per_system = _rows_per_system_for(strategy_name, strategy_overrides, count)
        if solution_rows_total is not None and strategy_name in cyclic_solution_strategies:
            # Binding b draws rows (b + p) mod K, so each binding takes the same count.
            capped = _solution_file_count(strategy_name, count, solution_rows_total)
            for binding_idx in range(len(bindings)):
                counts_by_binding[binding_idx][strategy_name] = capped
            continue
        if strategy_name in archive_globs:
            counts, files = _allocate_archive_rows(
                strategy_name=strategy_name,
                glob_pattern=archive_globs[strategy_name],
                skip=_archive_skip(strategy_overrides, strategy_name),
                rows=count,
                rows_per_system=rows_per_system,
                groups=groups,
                num_bindings=len(bindings),
            )
            for binding_idx, binding_count in enumerate(counts):
                if binding_count > 0:
                    counts_by_binding[binding_idx][strategy_name] = binding_count
                    files_by_binding[binding_idx][strategy_name] = files[binding_idx]
            continue
        if count == ALL_SAMPLES:
            if solution_rows_total is None:
                raise ValueError(
                    f"Strategy '{strategy_name}' requested samples=-1 (\"all\") across "
                    f"{len(bindings)} matrix bindings, but has no "
                    f"{_glob_key_phrase(strategy_name)} or solution_path to resolve a real "
                    "total from. Replicating -1 to every binding would load the full archive "
                    "once per binding (a cartesian-product blowup) — set an explicit positive "
                    "'samples' count instead."
                )
            # Every binding receives all rows of the solution source, so no split applies.
            for binding_idx in range(len(bindings)):
                counts_by_binding[binding_idx][strategy_name] = solution_rows_total
            continue
        allocations = _split_generated_rows(
            count, rows_per_system, groups, len(bindings), mixture.seed
        )
        for binding_idx, binding_count in enumerate(allocations):
            if binding_count > 0:
                counts_by_binding[binding_idx][strategy_name] = binding_count
    return BindingAllocation(
        counts=tuple(counts_by_binding),
        file_indices=tuple(files_by_binding),
    )


def _is_solution_archive(strategy_name: str) -> bool:
    """Whether a strategy draws from a glob of solution vectors."""
    props = _properties_for(strategy_name)
    return props is not None and props.is_archive and props.glob_key == "solutions_glob"


def _explicit_solution_strategies(strategy_counts: Mapping[str, int]) -> frozenset[str]:
    """Solution-archive strategies with an explicit positive count.

    Their per-binding rows come from an explicit solution file by the cyclic map.
    """
    return frozenset(
        name for name, count in strategy_counts.items() if count > 0 and _is_solution_archive(name)
    )


def _explicit_solution_rows_requested(strategy_counts: Mapping[str, int]) -> bool:
    """Whether an explicit solution file supplies rows by position rather than by id.

    True when any strategy asks for ``samples=-1`` or for an explicit solution-archive count.
    """
    return ALL_SAMPLES in strategy_counts.values() or bool(
        _explicit_solution_strategies(strategy_counts)
    )


def _solution_file_count(strategy_name: str, count: int, file_rows: int) -> int:
    """Rows one binding draws from an explicit solution file of ``file_rows`` rows.

    ``samples=-1`` takes every row. An explicit count is capped at the file's row count,
    since one binding cannot draw the same row twice; the cap is reported once.
    """
    if count == ALL_SAMPLES:
        return file_rows
    if count <= file_rows:
        return count
    logger.warning(
        f"Strategy '{strategy_name}' requested {count} rows from a solution file with "
        f"{file_rows} rows: emitting {file_rows} rows per binding."
    )
    return file_rows


def _merge_binding_file_overrides(
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    strategy_files: Mapping[str, tuple[int, ...]] | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    """Layer each strategy's explicit per-binding archive file indices onto its overrides.

    Never mutates ``strategy_overrides`` — it's shared across every binding's call.
    """
    if not strategy_files:
        return strategy_overrides
    merged = {name: dict(opts) for name, opts in (strategy_overrides or {}).items()}
    for strategy_name, file_indices in strategy_files.items():
        merged.setdefault(strategy_name, {})["file_indices"] = file_indices
    return merged


@dataclass(frozen=True)
class _GenerationRunContext:
    """Opened streams plus the per-binding budgets resolved against them.

    Attributes:
        streams: Every opened source stream and the resolved bindings.
        allocation: Per-binding strategy count and archive file-index maps.
        solution_by_position: Whether the explicit solution file's rows feed the bindings by
            position. Its rows are read per binding, never all at once.
        cyclic_solution_strategies: Strategies whose explicit count draws the cyclic
            rows ``(b + p) mod K`` of ``solution_rows`` on binding ``b``.
    """

    streams: OpenedStreams
    allocation: BindingAllocation
    solution_by_position: bool = False
    cyclic_solution_strategies: frozenset[str] = frozenset()


def _solution_row_total(solution_path: str | None, stream: VectorSampleStream | None) -> int | None:
    """Rows the solution source supplies by position: header-sized file or matched file count.

    An explicit file is sized from its header; a glob is sized by the number of files it
    matched, which the stream already lists. No content is read for either.
    """
    if solution_path is None or stream is None:
        return None
    path = Path(solution_path)
    if path.is_file():
        return solution_row_count(path)
    return len(stream.sample_ids)


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
    strategy_counts = resolve_strategy_counts(
        spec.mixture.counts, spec.mixture.mix, spec.mixture.total
    )
    rows_by_position = source.solution_path is not None and _explicit_solution_rows_requested(
        strategy_counts
    )
    streams = _open_streams(source, solution_unbound=rows_by_position)
    solution_rows_total = (
        _solution_row_total(source.solution_path, streams.solution) if rows_by_position else None
    )
    allocation = _resolve_binding_strategy_counts(
        bindings=streams.bindings,
        spec=spec,
        num_matrix_samples=len(streams.matrix.sample_ids),
        has_solution_source=streams.solution is not None,
        solution_rows_total=solution_rows_total,
    )
    return _GenerationRunContext(
        streams=streams,
        allocation=allocation,
        solution_by_position=rows_by_position and streams.solution is not None,
        cyclic_solution_strategies=_explicit_solution_strategies(strategy_counts),
    )


def _solution_archive_rows(
    context: _GenerationRunContext,
    binding_index: int,
    strategy_name: str,
    count: int,
) -> np.ndarray | None:
    """The solution rows one strategy on one binding receives, in draw order.

    Only solution-archive strategies receive archive rows. A generated strategy never does,
    whatever its count, because the explicit file is not a sample of its distribution.
    A cyclic strategy draws ``count`` rows starting at ``binding_index`` (``(b + p) mod K``);
    any other archive strategy receives the whole explicit file. Rows are read here, per
    binding, so the file is never held whole for the run.
    """
    stream = context.streams.solution
    if (
        not context.solution_by_position
        or stream is None
        or not _is_solution_archive(strategy_name)
    ):
        return None
    sample_ids = stream.sample_ids
    total = len(sample_ids)
    if total == 0:
        raise ValueError(
            f"Strategy '{strategy_name}' draws from an explicit solution file that has no samples."
        )
    if strategy_name in context.cyclic_solution_strategies:
        positions = [(binding_index + p) % total for p in range(count)]
    else:
        positions = list(range(total))
    return np.stack(
        [
            np.asarray(stream.load_sample(sample_ids[pos]).vector, dtype=np.float64)
            for pos in positions
        ]
    )


@dataclass(frozen=True)
class _BindingInputs:
    """Everything one binding contributes to generation, loaded once per binding.

    Attributes:
        binding: The binding's matrix, RHS, solution and parameter sample ids.
        mixture: The global mixture narrowed to this binding (seed, counts, archive files).
        matrix: The binding's cached matrix and its normalization scale.
        single_rhs: The binding's explicit RHS sample, scaled to the matrix, or None.
        single_solution: The binding's explicit solution sample, or None.
        parameter_vectors: One vector per parameter stream (None when the binding has no sample).
    """

    binding: SystemBinding
    mixture: MixtureSpec
    matrix: _CachedMatrix
    single_rhs: np.ndarray | None
    single_solution: np.ndarray | None
    parameter_vectors: tuple[np.ndarray | None, ...]


def _binding_mixture(
    context: _GenerationRunContext, mixture: MixtureSpec, binding_index: int
) -> MixtureSpec:
    """Narrow the global mixture to one binding: its seed, counts and archive file indices."""
    return replace(
        mixture,
        seed=derive_seed(mixture.seed, "binding", binding_index),
        counts=context.allocation.counts[binding_index],
        mix=None,
        total=None,
        strategy_overrides=_merge_binding_file_overrides(
            mixture.strategy_overrides, context.allocation.file_indices[binding_index]
        ),
    )


def _load_binding_inputs(
    binding_index: int,
    context: _GenerationRunContext,
    get_matrix: Callable[[int], _CachedMatrix],
    mixture: MixtureSpec,
) -> _BindingInputs:
    """Load one binding's RHS, solution, parameter vectors and matrix from its streams."""
    streams = context.streams
    binding = streams.bindings[binding_index]
    cached = get_matrix(binding.matrix_sample_id)

    single_rhs: np.ndarray | None = None
    if streams.rhs is not None and binding.rhs_sample_id is not None:
        rhs_sample = streams.rhs.load_sample(binding.rhs_sample_id)
        single_rhs = np.asarray(rhs_sample.vector, dtype=np.float64)
        if single_rhs.shape[0] != cached.matrix_norm.shape[0]:
            raise ValueError(
                f"RHS sample {binding.rhs_sample_id} length {single_rhs.shape[0]} "
                f"doesn't match matrix size {cached.matrix_norm.shape[0]}"
            )
        if cached.scale is not None:
            single_rhs = cached.scale.scale_rhs(single_rhs)

    single_solution: np.ndarray | None = None
    if streams.solution is not None and binding.solution_sample_id is not None:
        sol_sample = streams.solution.load_sample(binding.solution_sample_id)
        single_solution = np.asarray(sol_sample.vector, dtype=np.float64)

    parameter_vectors = tuple(
        np.asarray(stream.load_sample(sample_id).vector, dtype=np.float64)
        if sample_id is not None
        else None
        for stream, sample_id in zip(streams.parameters, binding.parameters_sample_ids, strict=True)
    )
    return _BindingInputs(
        binding=binding,
        mixture=_binding_mixture(context, mixture, binding_index),
        matrix=cached,
        single_rhs=single_rhs,
        single_solution=single_solution,
        parameter_vectors=parameter_vectors,
    )


def _make_strategy_runner(
    context: _GenerationRunContext,
    get_matrix: Callable[[int], _CachedMatrix],
    mixture: MixtureSpec,
    on_inputs_loaded: Callable[[int, _CachedMatrix], None],
) -> StrategyRunner:
    """Build the runner that generates one strategy's output for one binding.

    Bindings are visited one at a time, so only the most recent binding's inputs are kept.
    ``on_inputs_loaded`` receives each binding's matrix as it loads, so its scalars are
    available without loading the matrix again.
    """

    @lru_cache(maxsize=1)
    def _inputs_for(binding_index: int) -> _BindingInputs:
        logger.info(f"Generating/loading samples for binding index={binding_index}...")
        inputs = _load_binding_inputs(binding_index, context, get_matrix, mixture)
        on_inputs_loaded(binding_index, inputs.matrix)
        return inputs

    def run_strategy(binding_index: int, strategy_name: str) -> SampleBatch:
        inputs = _inputs_for(binding_index)
        n_features = inputs.matrix.matrix_norm.shape[0]
        count = context.allocation.counts[binding_index][strategy_name]
        strategy_rows = _generate_strategy_rows(
            inputs.matrix.matrix_norm,
            inputs.mixture,
            strategy_name,
            count,
            archive_solutions=_solution_archive_rows(context, binding_index, strategy_name, count),
            single_rhs=inputs.single_rhs,
            single_solution=inputs.single_solution,
        )
        if strategy_rows is None:
            strategy_rows = _StrategyRows(
                rhs=np.empty((0, n_features), dtype=np.float64),
                solutions=np.empty((0, n_features), dtype=np.float64),
                row_kind_codes=np.empty((0,), dtype=np.uint8),
            )
        row_count = strategy_rows.rhs.shape[0]
        return SampleBatch(
            binding_index=binding_index,
            strategy_name=strategy_name,
            rhs=np.asarray(strategy_rows.rhs, dtype=np.float64),
            solutions=np.asarray(strategy_rows.solutions, dtype=np.float64),
            row_kind_codes=np.asarray(strategy_rows.row_kind_codes, dtype=np.uint8),
            matrix_sample_index=np.full(row_count, inputs.binding.matrix_sample_id, dtype=np.int64),
            parameter_vectors=inputs.parameter_vectors,
        )

    return run_strategy


def _cached_matrix_loader(
    matrix_stream: MatrixSampleStream,
    spec: DatasetSpec,
    matrix_format: MatrixFormat,
) -> Callable[[int], _CachedMatrix]:
    """Return a loader that normalizes and measures one matrix sample per call.

    Only the most recent sample stays cached. Bindings are visited in order, so a matrix
    is needed only while its own binding runs, and this keeps one matrix in memory at a time.

    Args:
        matrix_stream: Source of the raw matrix samples.
        spec: Supplies the normalization and matrix norm settings.
        matrix_format: Storage format the raw matrices are loaded in.

    Returns:
        Callable mapping a matrix sample id to its cached normalized data.
    """

    @lru_cache(maxsize=1)
    def _get_matrix(sample_id: int) -> _CachedMatrix:
        raw_matrix = matrix_stream.load_sample(sample_id, matrix_format)
        if raw_matrix.shape[0] != raw_matrix.shape[1]:
            raise ValueError(f"Matrix sample {sample_id} must be square, got {raw_matrix.shape}")
        matrix_norm, scale, matrix_value_scale = normalize_matrix_for_generation(
            raw_matrix,
            spec.normalize,
            spectral_radius_bound=None,
        )
        matrix_norm_value = system_matrix_norm(matrix_norm, MatrixNormType(spec.matrix_norm_type))
        scale_params = serialize_scale_metadata(scale)
        return _CachedMatrix(
            matrix_norm=matrix_norm,
            scale=scale,
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )

    return _get_matrix


@dataclass(frozen=True)
class BatchStream:
    """Exact row plan and lazily generated batches of one dense generation run.

    Attributes:
        plan: Row budget of the run. Its total is the row count the writer must reach, and
            it is only defined when the plan is exact.
        single_matrix: True when every row shares one matrix (broadcast layout).
        batches: Batches in binding order, then strategy order. Consuming them also
            observes each binding's scalars into ``scale``, so ``scale`` is complete
            once the iterator is exhausted.
        matrix_for: Dense normalized matrix for a matrix sample id.
        scale: Cross-binding scalar aggregator fed by ``batches``.
    """

    plan: BatchPlan
    single_matrix: bool
    batches: Iterator[SampleBatch]
    matrix_for: Callable[[int], SystemMatrix]
    scale: ScalarAggregator


def open_batch_stream(
    source: SourceSpec,
    spec: DatasetSpec,
    *,
    batch_size: int,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> BatchStream:
    """Plan a generation run and expose its batches without buffering them.

    The stream does not persist anything; the caller writes the batches and reads
    ``scale`` afterwards. ``matrix_for`` returns the normalized matrix in ``matrix_format``.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: Strategy budgets, RNG controls, replacement policy and normalization.
        batch_size: Maximum rows per yielded batch.
        matrix_format: Storage format of the normalized matrices handed to ``matrix_for``.

    Returns:
        BatchStream with an unconsumed batch iterator. Its plan is exact only when every
        strategy count is concrete; an open-ended count has no total to pre-size from,
        and the caller must fall back to buffered generation.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    context = _prepare_generation_context(source, spec)
    streams = context.streams
    get_matrix = _cached_matrix_loader(streams.matrix, spec, matrix_format)
    plan = plan_batches(context.allocation)
    scale = ScalarAggregator()

    def matrix_for(sample_id: int) -> SystemMatrix:
        return get_matrix(sample_id).matrix_norm

    planned_indices = tuple(binding.binding_index for binding in plan.bindings)
    pending_scales: dict[int, BindingScale] = {}

    def record_scale(binding_index: int, cached: _CachedMatrix) -> None:
        pending_scales[binding_index] = BindingScale(
            matrix_norm_value=cached.matrix_norm_value,
            matrix_value_scale=cached.matrix_value_scale,
            scale_params=cached.scale_params,
        )

    def observe_binding(binding_index: int) -> None:
        scale.observe(pending_scales.pop(binding_index))

    def observed_batches() -> Iterator[SampleBatch]:
        # Batches arrive in binding order, so every planned binding up to the current one is
        # finished. Observing them in plan order keeps each planned binding observed exactly
        # once, including one whose strategies emit no batch at all.
        next_planned = 0

        def observe_through(binding_index: int | None) -> None:
            nonlocal next_planned
            while next_planned < len(planned_indices) and (
                binding_index is None or planned_indices[next_planned] <= binding_index
            ):
                observe_binding(planned_indices[next_planned])
                next_planned += 1

        for batch in generate_batches(
            plan,
            _make_strategy_runner(context, get_matrix, spec.mixture, record_scale),
            batch_size=batch_size,
        ):
            observe_through(batch.binding_index)
            yield batch
        observe_through(None)

    return BatchStream(
        plan=plan,
        single_matrix=streams.single_matrix_mode,
        batches=observed_batches(),
        matrix_for=matrix_for,
        scale=scale,
    )


__all__ = [
    "BatchStream",
    "BindingAllocation",
    "open_batch_stream",
]
