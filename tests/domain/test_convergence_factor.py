"""The average per-iteration residual reduction of each system, from its stored residual history."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.solver.reporting.convergence_factor import (
    CONVERGENCE_FACTOR_SCHEMA,
    convergence_factors,
)
from neuralls.domain.solver.reporting.event_frames import CONVERGENCE_SCHEMA


def _history(
    system_id: str, residuals: list[float], method: str = "amg"
) -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "system_id": system_id,
            "iteration": iteration,
            "residual_rel": residual,
            "energy_error_rel": None,
        }
        for iteration, residual in enumerate(residuals)
    ]


@pytest.fixture
def convergence() -> pl.DataFrame:
    """Histories of five systems, stored out of iteration order.

    ``halving`` halves its residual each of 3 iterations, ``uneven`` falls from 0.5 to 0.005 in
    2, ``exact`` reaches a zero residual, ``unstarted`` has only its initial residual and ``zero``
    starts at a zero residual (a zero right-hand side).
    """
    rows = [
        *_history("halving", [1.0, 0.5, 0.25, 0.125]),
        *_history("uneven", [0.5, 0.4, 0.005]),
        *_history("exact", [1.0, 0.0]),
        *_history("unstarted", [1.0]),
        *_history("zero", [0.0, 0.0]),
    ]
    return pl.DataFrame(rows[::-1], schema=CONVERGENCE_SCHEMA)


def _factors(frame: pl.DataFrame) -> dict[str, float | None]:
    return dict(zip(frame["system_id"], frame["convergence_factor"], strict=True))


def test_the_factors_have_one_row_per_system_with_the_schema(convergence: pl.DataFrame) -> None:
    frame = convergence_factors(convergence)

    assert frame.schema == pl.Schema(CONVERGENCE_FACTOR_SCHEMA)
    assert sorted(frame["system_id"]) == ["exact", "halving", "uneven", "unstarted", "zero"]


def test_the_factor_is_the_geometric_mean_reduction_over_the_iterations(
    convergence: pl.DataFrame,
) -> None:
    factors = _factors(convergence_factors(convergence))

    assert factors["halving"] == pytest.approx(0.5)
    assert factors["uneven"] == pytest.approx(0.1)


def test_a_zero_final_residual_has_a_zero_factor(convergence: pl.DataFrame) -> None:
    assert _factors(convergence_factors(convergence))["exact"] == 0.0


def test_a_history_without_an_iteration_or_a_nonzero_start_has_no_factor(
    convergence: pl.DataFrame,
) -> None:
    factors = _factors(convergence_factors(convergence))

    assert (factors["unstarted"], factors["zero"]) == (None, None)


def test_the_factors_of_several_methods_stay_apart() -> None:
    convergence = pl.DataFrame(
        [*_history("s", [1.0, 0.5]), *_history("s", [1.0, 0.25], method="jacobi")],
        schema=CONVERGENCE_SCHEMA,
    )

    frame = convergence_factors(convergence).sort("method")

    assert frame.rows() == [("amg", "s", 0.5), ("jacobi", "s", 0.25)]


def test_an_empty_history_frame_has_no_factor() -> None:
    frame = convergence_factors(pl.DataFrame(schema=CONVERGENCE_SCHEMA))

    assert (frame.height, frame.schema) == (0, pl.Schema(CONVERGENCE_FACTOR_SCHEMA))
