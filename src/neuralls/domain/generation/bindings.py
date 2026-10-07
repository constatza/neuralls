"""Pure ID-level binding across matrix/rhs/parameters/solution sample streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemBinding:
    """ID-level binding across matrix/rhs/parameters/solution sources."""

    sample_id: int
    matrix_sample_id: int
    rhs_sample_id: int | None = None
    parameters_sample_ids: tuple[int | None, ...] = ()
    solution_sample_id: int | None = None


def bind_sources(
    matrix_ids: tuple[int, ...],
    rhs_ids: tuple[int, ...] | None = None,
    solution_ids: tuple[int, ...] | None = None,
    parameters_ids_list: tuple[tuple[int, ...], ...] = (),
) -> list[SystemBinding]:
    """Bind matrix/rhs/solution/parameters ids with single-matrix broadcast semantics.

    Rules:
    - If only one matrix id exists and vectors have many ids, broadcast matrix id.
    - Otherwise bindings are keyed by matrix ids.
    - Provided rhs ids must match matrix ids (except single-matrix broadcast).
    - Provided solution ids must match matrix ids (except single-matrix broadcast).
    - Each entry in parameters_ids_list is a tuple of sample IDs for one parameter stream.

    Args:
        matrix_ids: Sample IDs from the matrix stream.
        rhs_ids: Optional sample IDs from the RHS stream.
        solution_ids: Optional sample IDs from the solution stream.
        parameters_ids_list: Tuple of ID tuples, one per parameter stream.

    Returns:
        List of ``SystemBinding`` objects with all sources resolved.
    """
    if not matrix_ids:
        raise ValueError("No matrix samples available to bind.")

    matrix_set = set(matrix_ids)
    rhs_set = set(rhs_ids or ())
    solution_set = set(solution_ids or ())
    param_sets = [set(ids) for ids in parameters_ids_list]
    _check_parameter_ids(matrix_set, param_sets)

    if len(matrix_set) == 1:
        matrix_sample_id = next(iter(matrix_set))
        candidate_ids = (
            rhs_set if rhs_set else (solution_set if solution_set else {matrix_sample_id})
        )
        bound_ids = sorted(candidate_ids)
        return [
            _binding(sample_id, matrix_sample_id, rhs_set, solution_set, len(param_sets))
            for sample_id in bound_ids
        ]

    bound_ids = sorted(matrix_set)
    if rhs_set and rhs_set != matrix_set:
        missing = sorted(matrix_set - rhs_set)
        extra = sorted(rhs_set - matrix_set)
        raise ValueError(
            f"RHS IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    if solution_set and solution_set != matrix_set:
        missing = sorted(matrix_set - solution_set)
        extra = sorted(solution_set - matrix_set)
        raise ValueError(
            f"solution IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    return [
        _binding(sample_id, sample_id, rhs_set, solution_set, len(param_sets))
        for sample_id in bound_ids
    ]


def _check_parameter_ids(matrix_set: set[int], param_sets: Sequence[set[int]]) -> None:
    """Require every parameter stream to hold exactly the matrix sample ids.

    Parameters describe a matrix, so each matrix sample has one parameter sample and no
    other. A missing or extra id is an error rather than a silently unbound sample.

    Raises:
        ValueError: If a parameter stream's ids differ from the matrix ids.
    """
    for index, ids in enumerate(param_sets):
        if ids == matrix_set:
            continue
        missing = sorted(matrix_set - ids)
        extra = sorted(ids - matrix_set)
        raise ValueError(
            f"parameter stream {index} IDs must match matrix IDs. Missing={missing}, extra={extra}"
        )


def _binding(
    sample_id: int,
    matrix_sample_id: int,
    rhs_set: set[int],
    solution_set: set[int],
    parameter_stream_count: int,
) -> SystemBinding:
    """Bind one sample; every parameter stream takes the matrix's own sample id."""
    return SystemBinding(
        sample_id=sample_id,
        matrix_sample_id=matrix_sample_id,
        rhs_sample_id=sample_id if sample_id in rhs_set else None,
        solution_sample_id=sample_id if sample_id in solution_set else None,
        parameters_sample_ids=(matrix_sample_id,) * parameter_stream_count,
    )
