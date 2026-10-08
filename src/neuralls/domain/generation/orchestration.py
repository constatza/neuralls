"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from neuralls.shared.types import MatrixFormat, SystemMatrix

from .archive_files import solution_row_count
from .batch import SampleBatch
from .batch_generator import StrategyRunner, generate_batches
from .batch_plan import ALL_SAMPLES, BatchPlan, BindingAllocation, plan_batches
from .binding_allocation import _resolve_binding_strategy_counts
from .bindings import SystemBinding, bind_sources
from .counts import resolve_strategy_counts
from .file_sources import MatrixReaders
from .matrix_cache import _cached_matrix_loader, _CachedMatrix
from .scalar_aggregate import BindingScale, ScalarAggregator
from .seeds import derive_seed
from .single_slot import SingleSlot
from .source_streams import (
    MatrixSampleStream,
    VectorSampleStream,
    open_matrix_stream,
    open_vector_stream,
)
from .specs import DatasetSpec, MixtureSpec, SourceSpec
from .strategy_properties import _explicit_solution_strategies, _is_solution_archive
from .strategy_rows import _generate_strategy_rows, _StrategyRows


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


def _open_streams(
    source: SourceSpec, *, readers: MatrixReaders, solution_unbound: bool = False
) -> OpenedStreams:
    """Open matrix, optional RHS, optional solution, and optional parameter streams and bind them.

    Args:
        source: Resolved source paths and per-stream sample filters.
        readers: Reads raw arrays/matrices at a path; injected so this domain
            module never depends on platform's I/O directly.
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
            readers=readers,
        )

    matrix_stream = open_matrix_stream(
        matrix_path_expr=source.matrix_path,
        sample_id_regex=source.sample_id_regex,
        enumerate_by=source.enumerate_by,
        include_indices=source.include_indices,
        exclude_indices=source.exclude_indices,
        readers=readers,
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


def _explicit_solution_rows_requested(strategy_counts: Mapping[str, int]) -> bool:
    """Whether an explicit solution file supplies rows by position rather than by id.

    True when any strategy asks for ``samples=-1`` or for an explicit solution-archive count.
    """
    return ALL_SAMPLES in strategy_counts.values() or bool(
        _explicit_solution_strategies(strategy_counts)
    )


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
    *,
    readers: MatrixReaders,
) -> _GenerationRunContext:
    """Open all source streams, bind them, and resolve per-binding strategy counts.

    Args:
        source: Resolved source paths and per-stream sample filters.
        spec: Dataset assembly settings supplying the strategy budgets.
        readers: Reads raw arrays/matrices at a path; injected so this domain
            module never depends on platform's I/O directly.

    Returns:
        _GenerationRunContext pairing the opened streams with their allocation.
    """
    strategy_counts = resolve_strategy_counts(
        spec.mixture.counts, spec.mixture.mix, spec.mixture.total
    )
    rows_by_position = source.solution_path is not None and _explicit_solution_rows_requested(
        strategy_counts
    )
    streams = _open_streams(source, readers=readers, solution_unbound=rows_by_position)
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
    slot = SingleSlot[int, _BindingInputs]()

    def _load(binding_index: int) -> _BindingInputs:
        logger.info(f"Generating/loading samples for binding index={binding_index}...")
        inputs = _load_binding_inputs(binding_index, context, get_matrix, mixture)
        on_inputs_loaded(binding_index, inputs.matrix)
        return inputs

    def _inputs_for(binding_index: int) -> _BindingInputs:
        return slot.get(binding_index, _load)

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
    readers: MatrixReaders,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> BatchStream:
    """Plan a generation run and expose its batches without buffering them.

    The stream does not persist anything; the caller writes the batches and reads
    ``scale`` afterwards. ``matrix_for`` returns the normalized matrix in ``matrix_format``.

    Args:
        source: Where the run reads its matrix/RHS/solution/parameter samples from.
        spec: Strategy budgets, RNG controls, replacement policy and normalization.
        batch_size: Maximum rows per yielded batch.
        readers: Reads raw arrays/matrices at a path; injected so this domain
            module never depends on platform's I/O directly.
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
    context = _prepare_generation_context(source, spec, readers=readers)
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
