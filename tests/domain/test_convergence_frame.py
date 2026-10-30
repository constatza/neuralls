"""Per-iteration convergence table: one row per residual iteration, energy null when absent."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    build_convergence_frame,
    validate_decoded_convergence_frame,
)
from neuralls.shared.types import PreconditionerName, SystemId
from tests.convergence_support import cg_result

METHOD = PreconditionerName("jacobi")


def test_energy_is_null_where_the_energy_history_is_missing_or_short() -> None:
    frame = build_convergence_frame(
        METHOD,
        (
            (SystemId("with"), cg_result([1.0, 0.1, 0.01], [1.0, 0.2])),
            (SystemId("without"), cg_result([1.0, 0.5])),
        ),
    )

    assert frame.schema == pl.Schema(CONVERGENCE_SCHEMA)
    assert frame.select("system_id", "iteration", "energy_error_rel").rows() == [
        ("with", 0, 1.0),
        ("with", 1, 0.2),
        ("with", 2, None),
        ("without", 0, None),
        ("without", 1, None),
    ]


def test_a_repeated_method_system_iteration_is_rejected() -> None:
    frame = build_convergence_frame(METHOD, ((SystemId("s"), cg_result([1.0, 0.5])),))

    with pytest.raises(ValueError, match="duplicate convergence row"):
        validate_decoded_convergence_frame(pl.concat([frame, frame]))
