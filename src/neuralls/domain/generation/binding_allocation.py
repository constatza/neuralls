"""Per-binding strategy-count and archive-file-index allocation, pure given a seed."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from .allocation import archive_units, split_remainder
from .archive_files import select_archive_files
from .batch_plan import ALL_SAMPLES, BindingAllocation
from .counts import resolve_strategy_counts
from .runner import rows_per_base_system
from .seeds import derive_seed
from .source_streams import SystemBinding
from .specs import DatasetSpec
from .strategy_properties import (
    _archive_glob_for_strategy,
    _explicit_solution_strategies,
    _properties_for,
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
