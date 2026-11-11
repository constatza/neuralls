"""Extra residual measurements: the true residual of a solve and the exact-reference outcome.

These are additive facts stored next to the solver's own outcome; none of them changes how a
solve is classified.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch
from torchalg import pcg

from neuralls.domain.solver.comparison import compute_reference_solution, prepare_reference
from neuralls.domain.solver.error_metrics import ReferenceSolver
from neuralls.domain.solver.models.benchmark_events import (
    ExactReference,
    ExactSolutionSolved,
    ExactSolutionUnavailable,
    MeasuredPeakMemory,
    MeasuredResidual,
    ResidualMeasurement,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnmeasuredResidual,
)
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.reporting.event_frames import (
    build_solve_frame,
    decode_solve_frame,
    validate_decoded_solve_frame,
)
from neuralls.domain.solver.true_residual import (
    NonFiniteResidual,
    measure_residual,
    shared_threshold,
    stopping_threshold_rel,
)
from neuralls.platform.reporting.method_figures import residual_threshold
from neuralls.shared.device import SingleRun
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId
from tests.provenance_support import provenance_for
from tests.recipe_support import GAUSSIAN


def _attempt(
    system: str,
    *,
    outcome: SolveConverged | SolveFailed,
    residual: ResidualMeasurement,
    exact_reference: ExactReference,
) -> SolveAttempt:
    return SolveAttempt(
        setup_event_id=SetupEventId("e0"),
        system_id=SystemId(system),
        dataset=DatasetId("ds"),
        recipe=GAUSSIAN,
        matrix_id=MatrixId(0),
        rhs_ordinal=0,
        method=PreconditionerName("jacobi"),
        solve_s=0.1,
        samples_s=(0.1,),
        peak_memory=MeasuredPeakMemory(5),
        outcome=outcome,
        residual=residual,
        exact_reference=exact_reference,
        repeats=SingleRun(),
    )


@pytest.fixture
def attempts() -> tuple[SolveAttempt, ...]:
    """Every combination of measurement state a stored attempt can carry."""
    return (
        _attempt(
            "s0",
            outcome=SolveConverged(7, 1e-9),
            residual=MeasuredResidual(rhs_norm=2.0, true_residual_rel=3e-9),
            exact_reference=ExactSolutionSolved(true_residual_rel=1e-12),
        ),
        _attempt(
            "s1",
            outcome=SolveFailed(0, "solver raised"),
            residual=UnmeasuredResidual(),
            exact_reference=ExactSolutionUnavailable("iteration budget exhausted"),
        ),
    )


def test_measurements_survive_a_frame_round_trip(attempts: tuple[SolveAttempt, ...]) -> None:
    assert decode_solve_frame(build_solve_frame(attempts)) == attempts


def test_a_recursively_converged_solve_may_carry_a_larger_true_residual(
    attempts: tuple[SolveAttempt, ...],
) -> None:
    """The extra measurement never reclassifies or rejects the solver's own outcome."""
    converged = replace(attempts[0], residual=MeasuredResidual(rhs_norm=2.0, true_residual_rel=1.0))

    assert decode_solve_frame(build_solve_frame((converged,))) == (converged,)


@pytest.mark.parametrize(
    "column, value",
    [
        ("true_residual_rel", None),
        ("rhs_norm", None),
        ("reference_true_residual_rel", None),
    ],
)
def test_measured_states_reject_a_missing_payload(
    attempts: tuple[SolveAttempt, ...], column: str, value: object
) -> None:
    frame = build_solve_frame(attempts[:1]).with_columns(
        pl.lit(value, dtype=pl.Float64).alias(column)
    )

    with pytest.raises(ValueError):
        validate_decoded_solve_frame(frame)


@pytest.mark.parametrize(
    "column, value",
    [
        ("true_residual_rel", 1e-3),
        ("rhs_norm", 1.0),
        ("reference_true_residual_rel", 1e-3),
    ],
)
def test_unmeasured_states_reject_a_payload(
    attempts: tuple[SolveAttempt, ...], column: str, value: float
) -> None:
    frame = build_solve_frame(attempts[1:]).with_columns(
        pl.lit(value, dtype=pl.Float64).alias(column)
    )

    with pytest.raises(ValueError):
        validate_decoded_solve_frame(frame)


def test_an_unavailable_reference_must_say_why(attempts: tuple[SolveAttempt, ...]) -> None:
    frame = build_solve_frame(attempts[1:]).with_columns(
        pl.lit(None, dtype=pl.String).alias("reference_reason")
    )

    with pytest.raises(ValueError):
        validate_decoded_solve_frame(frame)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_measured_values_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError):
        MeasuredResidual(rhs_norm=1.0, true_residual_rel=bad)
    with pytest.raises(ValueError):
        MeasuredResidual(rhs_norm=bad, true_residual_rel=1.0)
    with pytest.raises(ValueError):
        ExactSolutionSolved(true_residual_rel=bad)


def test_measure_residual_is_the_relative_norm_of_b_minus_ax(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, known_solution: torch.Tensor
) -> None:
    perturbed = known_solution + 1e-3

    measured = measure_residual(spd_matrix, rhs, perturbed)

    expected = float(
        torch.linalg.vector_norm(rhs - spd_matrix @ perturbed) / torch.linalg.vector_norm(rhs)
    )
    assert isinstance(measured, MeasuredResidual)
    assert measured.rhs_norm == pytest.approx(float(torch.linalg.vector_norm(rhs)))
    assert measured.true_residual_rel == pytest.approx(expected)


def test_measure_residual_of_a_zero_rhs_is_unmeasured(spd_matrix: torch.Tensor) -> None:
    zero = torch.zeros(spd_matrix.shape[0], dtype=spd_matrix.dtype)

    assert measure_residual(spd_matrix, zero, zero) == UnmeasuredResidual()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_measure_residual_of_a_non_finite_iterate_is_non_finite(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, bad: float
) -> None:
    iterate = torch.zeros_like(rhs)
    iterate[3] = bad

    assert measure_residual(spd_matrix, rhs, iterate) == NonFiniteResidual()


def test_measure_residual_of_a_zero_rhs_with_a_non_finite_iterate_is_non_finite(
    spd_matrix: torch.Tensor,
) -> None:
    zero = torch.zeros(spd_matrix.shape[0], dtype=spd_matrix.dtype)

    assert measure_residual(spd_matrix, zero, torch.full_like(zero, float("nan"))) == (
        NonFiniteResidual()
    )


def test_measure_residual_of_a_finite_iterate_whose_residual_overflows_is_non_finite(
    rhs: torch.Tensor,
) -> None:
    huge = torch.eye(rhs.shape[0], dtype=rhs.dtype) * torch.finfo(rhs.dtype).max
    iterate = torch.full_like(rhs, 10.0)

    assert measure_residual(huge, rhs, iterate) == NonFiniteResidual()


@pytest.mark.parametrize(
    "rtol, atol, rhs_norm, expected",
    [(1e-6, 1e-14, 2.0, 1e-6), (1e-6, 1e-3, 2.0, 5e-4)],
)
def test_stopping_threshold_is_the_larger_of_rtol_and_atol_over_rhs_norm(
    rtol: float, atol: float, rhs_norm: float, expected: float
) -> None:
    assert stopping_threshold_rel(rtol, atol, rhs_norm) == pytest.approx(expected)


def test_stopping_threshold_matches_what_the_solver_applies(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    """The reported line is the level at which the solver stops, in both tolerance regimes."""
    for rtol, atol in ((1e-6, 1e-14), (1e-12, 1e-3 * float(torch.linalg.vector_norm(rhs)))):
        _, info = pcg(spd_matrix, rhs, torch.zeros_like(rhs), rtol=rtol, atol=atol, maxiter=200)
        threshold = stopping_threshold_rel(rtol, atol, info.rhs_norm)

        assert info.converged
        assert info.residual <= threshold


def test_shared_threshold_needs_every_system_to_agree() -> None:
    assert shared_threshold([1e-6, 1e-6]) == 1e-6
    assert shared_threshold([1e-6, 2e-6]) is None
    assert shared_threshold([]) is None


def test_reference_solve_reports_its_achieved_residual(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    solution, reference = compute_reference_solution(
        prepare_reference(spd_matrix), rhs, rtol=1e-6, margin=1e-2
    )

    assert solution is not None
    assert isinstance(reference, ExactSolutionSolved)
    assert reference.true_residual_rel <= 1e-6 * 1e-2 * 10


def test_failed_reference_solve_keeps_its_reason(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    solution, reference = compute_reference_solution(
        prepare_reference(spd_matrix), rhs, rtol=1e-6, margin=2.0
    )

    assert solution is None
    assert isinstance(reference, ExactSolutionUnavailable)
    assert "reference_precision_margin" in reference.reason


def test_non_finite_reference_solution_is_unavailable_not_an_error(
    spd_matrix: torch.Tensor, rhs: torch.Tensor
) -> None:
    nan_solver = ReferenceSolver(spd_matrix, lambda b: np.full_like(b, np.nan))

    solution, reference = compute_reference_solution(nan_solver, rhs, rtol=1e-6, margin=1e-2)

    assert solution is None
    assert reference == ExactSolutionUnavailable(
        "Reference direct solve could not reach relative residual 1.0e-08; achieved nan."
    )


def test_failed_factorization_is_unavailable_for_every_rhs(rhs: torch.Tensor) -> None:
    singular = torch.zeros(rhs.numel(), rhs.numel(), dtype=torch.float64).to_sparse_csr()

    prepared = prepare_reference(singular)
    solution, reference = compute_reference_solution(prepared, rhs, rtol=1e-6, margin=1e-2)

    assert solution is None
    assert reference is prepared


@pytest.fixture
def two_systems() -> tuple[SystemProvenance, ...]:
    tags = tuple(
        SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(0), ordinal, SystemId(f"s{ordinal}"))
        for ordinal in range(2)
    )
    return provenance_for(tags)


def _stored(rhs_norms: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"system_id": ["s0", "s1"], "rhs_norm": rhs_norms},
        schema={"system_id": pl.String, "rhs_norm": pl.Float64},
    )


def test_figure_threshold_is_drawn_when_every_plotted_system_shares_it(
    two_systems: tuple[SystemProvenance, ...],
) -> None:
    line = residual_threshold(_stored([1.0, 1.0]), two_systems, ["s0", "s1"])

    assert line == pytest.approx(stopping_threshold_rel(1e-6, 1e-10, 1.0))


def test_figure_threshold_is_omitted_when_systems_differ_or_are_unmeasured(
    two_systems: tuple[SystemProvenance, ...],
) -> None:
    assert residual_threshold(_stored([1.0, 1e-6]), two_systems, ["s0", "s1"]) is None
    assert residual_threshold(_stored([1.0, None]), two_systems, ["s0", "s1"]) is None
